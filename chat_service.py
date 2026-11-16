"""Executes one conversation turn. Shared by the WebSocket handler and the evaluator,
so evaluation exercises exactly the code path that serves users."""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents import prompts
from agents.sales_graph import new_state
from agents.session_manager import Session
from config import Settings, settings as default_settings

log = logging.getLogger(__name__)

Emit = Callable[[str], Awaitable[None]]

CROSSSELL = {
    "gaming": [
        {"label": "+ Gaming Mouse", "msg": "What gaming mice do you recommend?"},
        {"label": "+ Gaming Headset", "msg": "Do you have gaming headset recommendations?"},
    ],
    "business": [
        {"label": "+ Docking Station", "msg": "What docking stations work with this laptop?"},
        {"label": "+ Extended Warranty", "msg": "Tell me about extended warranty options"},
    ],
    "creative": [
        {"label": "+ External Monitor", "msg": "What monitors pair well with this for creative work?"},
        {"label": "+ Stylus/Pen", "msg": "Do you recommend a stylus for creative work?"},
    ],
    "student": [
        {"label": "+ Laptop Bag", "msg": "What laptop bags do you recommend for students?"},
        {"label": "+ Student Discount", "msg": "Are there any student discounts available?"},
    ],
    "smb": [
        {"label": "+ Business Support", "msg": "What business support plans are available?"},
        {"label": "+ Bulk Pricing", "msg": "Do you offer bulk pricing for businesses?"},
    ],
}


def build_llm(cfg: Settings = default_settings, temperature: Optional[float] = None):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=cfg.ollama_model,
        base_url=cfg.ollama_base_url,
        temperature=cfg.temperature if temperature is None else temperature,
        reasoning=cfg.reasoning,
    )


def build_cta(category: Optional[str]) -> str:
    return json.dumps({
        "actions": [
            {"label": "View Full Specs", "action": "specs"},
            {"label": "Compare Models", "action": "compare"},
            {"label": "Talk to a Rep", "action": "rep"},
        ],
        "crosssell": CROSSSELL.get(category or "all", []),
    })


@dataclass
class TurnResult:
    action: str
    response: str
    category: str = "all"
    intent: Optional[str] = None
    products: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    llm_called: bool = False
    fallback: bool = False
    latency_s: float = 0.0
    ttft_s: Optional[float] = None       # time to first visible token (LLM turns only)
    output_tokens: Optional[int] = None
    tokens_per_s: Optional[float] = None

    def metrics_line(self) -> str:
        parts = [f"latency:{self.latency_s}s", f"route:{self.category}", f"type:{self.action}"]
        if self.llm_called:
            if self.ttft_s is not None:
                parts.append(f"ttft:{self.ttft_s}s")
            if self.output_tokens is not None:
                parts.append(f"tokens:{self.output_tokens}")
            if self.tokens_per_s is not None:
                parts.append(f"tok/s:{self.tokens_per_s}")
            parts.append("llm:fallback" if self.fallback else "llm:called")
        else:
            parts.append("llm:skipped")
        return "[METRICS] " + " | ".join(parts)


def _fallback_answer(products: list) -> str:
    if not products:
        return "Sorry — the assistant is not responding right now. Please try again in a moment."
    lines = [prompts.FALLBACK_INTRO, ""]
    for p in products[:3]:
        lines.append(f"- **{p['name']}** — ${p['price']} | {p['processor']} | {p['gpu']} | {p['weight_lbs']} lbs")
    return "\n".join(lines)


async def _stream_llm(llm, messages: list, emit: Emit, cfg: Settings, started: float) -> dict:
    """Stream the answer with a first-token and a total deadline."""
    text, ttft, n_chunks, usage_tokens = "", None, 0, None
    stream = llm.astream(messages).__aiter__()
    deadline = started + cfg.total_timeout_s
    while True:
        limit = cfg.first_token_timeout_s if ttft is None else max(0.01, deadline - time.monotonic())
        try:
            chunk = await asyncio.wait_for(stream.__anext__(), timeout=limit)
        except StopAsyncIteration:
            break
        usage = getattr(chunk, "usage_metadata", None)
        if usage and usage.get("output_tokens"):
            usage_tokens = usage["output_tokens"]
        token = chunk.content if isinstance(chunk.content, str) else ""
        if not token:
            continue            # reasoning / empty chunks are not shown and don't count as first token
        if ttft is None:
            ttft = time.monotonic() - started
        n_chunks += 1
        text += token
        await emit(token)
    return {"text": text, "ttft": ttft, "tokens": usage_tokens or n_chunks}


async def run_turn(session: Session, user_msg: str, graph, llm, emit: Emit,
                   cfg: Settings = default_settings) -> TurnResult:
    started = time.monotonic()
    session.turn_count += 1

    # The graph is synchronous (embedding + vector search); keep it off the event loop.
    state = await asyncio.to_thread(graph.invoke, new_state(session, user_msg))

    action = state["action"]
    result = TurnResult(
        action=action, response=state.get("response") or "", intent=state.get("intent"),
        category=state.get("category") or "all", products=state.get("matched_products") or [],
        notes=state.get("notes") or [],
    )

    if action == "llm":
        result.llm_called = True
        history = session.history[-cfg.history_window:] if cfg.history_window else []
        messages = [SystemMessage(content=state["system_prompt"])] + history + [HumanMessage(content=user_msg)]
        emitted = ""

        async def tracking_emit(token: str):
            nonlocal emitted
            emitted += token
            await emit(token)

        try:
            out = await _stream_llm(llm, messages, tracking_emit, cfg, started)
            result.response = out["text"]
            result.ttft_s = round(out["ttft"], 2) if out["ttft"] is not None else None
            result.output_tokens = out["tokens"]
            gen_time = (time.monotonic() - started) - (out["ttft"] or 0)
            if out["tokens"] and gen_time > 0:
                result.tokens_per_s = round(out["tokens"] / gen_time, 1)
            if not result.response.strip():
                raise RuntimeError("LLM returned an empty answer")
        except Exception as exc:  # timeout, connection refused, empty answer ...
            log.warning("LLM call failed (%s: %s); sending catalog fallback", type(exc).__name__, exc)
            result.fallback = True
            fallback = _fallback_answer(result.products)
            if emitted:
                fallback = "\n\n_(The response was cut off.)_\n\n" + fallback
            await emit(fallback)
            result.response = emitted + fallback
    else:
        await emit(result.response)

    result.latency_s = round(time.monotonic() - started, 2)

    if action in ("llm", "cache", "factual"):
        await emit(f"[CTA]{build_cta(result.category)}")
    await emit("[DONE]")
    if action not in ("direct", "clarify"):
        await emit(result.metrics_line())

    # History: every turn, with the assistant reply stored as an AIMessage.
    session.history.append(HumanMessage(content=user_msg))
    session.history.append(AIMessage(content=result.response))
    if result.products:
        session.last_product_ids = [p["id"] for p in result.products]
    session.metrics.append({"action": action, "latency_s": result.latency_s, "ttft_s": result.ttft_s})

    if action == "llm" and not result.fallback:
        from agents.catalog import catalog_version
        session.cache.store_entry(
            user_msg, result.response, result.category,
            context=(session.profile.signature(), catalog_version()),
            product_ids=tuple(session.last_product_ids),
        )
    return result
