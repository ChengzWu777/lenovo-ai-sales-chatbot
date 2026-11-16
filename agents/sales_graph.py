"""LangGraph decision pipeline: intent -> slots -> cache -> clarify -> retrieve.

The graph only decides what to do. It returns an action plus everything needed to
execute it; the LLM call itself is streamed by chat_service.run_turn.

Actions
  direct    canned reply (greeting, support, off-topic, accessories, hand-off)
  clarify   ask for use case / budget
  cache     reuse an earlier answer for the same question and profile
  factual   answered from the catalog, no LLM
  no_match  nothing satisfies the hard constraints; say so, no LLM
  llm       stream an answer grounded in state["matched_products"]
"""
import logging
from dataclasses import asdict
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents import prompts
from agents.catalog import catalog_version, find_products_by_name, get_by_id, load_products
from agents.filters import apply_constraints, budget_ceiling, passes_hard, violations
from agents.intent_classifier import Intent, classify_intent
from agents.product_matcher import (
    detect_superlative, format_factual_response, format_list_response,
    format_products_for_prompt, is_comparison, is_factual, is_list_request, rank_products,
)
from agents.session_manager import Session
from agents.slot_filler import extract_slots, format_profile_summary
from agents.taxonomy import infer_category
from agents.text import tokenize
from config import settings

log = logging.getLogger(__name__)

RETRIEVE_K = 12
PINNED_FOLLOWUP = 4      # products from the previous answer kept for follow-up questions
LIST_CAP = 15


class ChatState(TypedDict, total=False):
    session: Session
    user_msg: str
    intent: str
    action: str
    response: Optional[str]
    category: Optional[str]
    matched_products: Optional[list]
    system_prompt: Optional[str]
    notes: list
    slots_changed: bool
    skip: bool


def new_state(session: Session, user_msg: str) -> ChatState:
    return {
        "session": session, "user_msg": user_msg, "intent": None, "action": None,
        "response": None, "category": None, "matched_products": None,
        "system_prompt": None, "notes": [], "slots_changed": False, "skip": False,
    }


# Words that say "I want a laptop" without saying which kind.
_GENERIC = {
    "i", "a", "an", "the", "need", "want", "looking", "for", "to", "buy", "get", "purchase",
    "laptop", "laptops", "computer", "notebook", "pc", "lenovo", "new", "good", "nice", "great",
    "best", "recommend", "recommendation", "suggest", "something", "one", "me", "my", "can", "you",
    "help", "please", "what", "do", "would", "like", "am", "m", "is", "of", "some", "any", "show",
    "find", "pick", "choose", "which", "should", "have", "your", "and", "it", "that", "with",
}


def _is_vague(msg: str) -> bool:
    informative = [t for t in tokenize(msg) if t not in _GENERIC]
    return len(informative) < 2


def _cache_context(session: Session) -> tuple:
    return (session.profile.signature(), catalog_version())


def _hard_constraint_summary(profile) -> str:
    parts = []
    if profile.budget_max:
        parts.append(f"a budget of ${profile.budget_max}")
    if profile.processor_reject:
        parts.append(f"no {'/'.join(profile.processor_reject)} processors")
    if profile.brand_reject:
        parts.append(f"no {'/'.join(profile.brand_reject)}")
    return ", ".join(parts) or "your requirements"


def _no_match_response(profile) -> str:
    summary = _hard_constraint_summary(profile)
    # Closest alternative: keep every exclusion, drop only the budget.
    relaxed = [p for p in load_products()
               if not any(v.startswith(("uses an excluded", "is an excluded")) for v in violations(p, profile))]
    if profile.budget_max and relaxed:
        closest = min(relaxed, key=lambda p: p["price"])
        over = closest["price"] - profile.budget_max
        return (
            f"I don't have a laptop that fits {summary}. The closest option is the "
            f"**{closest['name']}** at **${closest['price']}** (${over} over your budget). "
            "Would you like to raise the budget or relax another requirement?"
        )
    return (f"I don't have a laptop that fits {summary}. "
            "Would you like to relax one of those requirements?")


def build_sales_graph(retriever="auto"):
    """retriever: "auto" loads the ChromaDB retriever lazily and falls back to
    lexical ranking if it is unavailable; None disables semantic ranking; any
    object with .rank(query, candidate_ids, n_results, use_case) can be injected."""
    holder = {"retriever": retriever, "warned": False}

    def semantic_rank(query: str, candidate_ids: list, use_case: Optional[str]) -> Optional[list]:
        r = holder["retriever"]
        if r is None:
            return None
        try:
            if r == "auto":
                from agents.rag_retriever import get_retriever
                r = holder["retriever"] = get_retriever()
            return r.rank(query, candidate_ids=candidate_ids, n_results=RETRIEVE_K, use_case=use_case)
        except Exception:
            if not holder["warned"]:
                log.exception("Semantic retrieval unavailable; falling back to lexical ranking")
                holder["warned"] = True
            if holder["retriever"] == "auto":
                holder["retriever"] = None
            return None

    # ── Nodes ──
    def node_intent(state: ChatState) -> ChatState:
        session = state["session"]
        has_context = (not session.profile.is_empty() or bool(session.last_product_ids)
                       or session.pending_clarify)
        session.pending_clarify = False
        intent = classify_intent(state["user_msg"], has_context=has_context)
        state["intent"] = intent.value

        direct = {
            Intent.GREETING: prompts.GREETING_RESPONSE,
            Intent.THANKS: prompts.THANKS_RESPONSE,
            Intent.SUPPORT_FAQ: prompts.SUPPORT_RESPONSE,
            Intent.OFF_TOPIC: prompts.OFF_TOPIC_RESPONSE,
            Intent.ACCESSORY: prompts.ACCESSORY_RESPONSE,
        }
        if intent in direct:
            state.update(action="direct", response=direct[intent], skip=True)
        elif intent == Intent.HANDOFF:
            state.update(
                action="direct", skip=True,
                response=prompts.HANDOFF_RESPONSE.format(
                    profile_summary=format_profile_summary(session.profile)),
            )
        return state

    def node_slots(state: ChatState) -> ChatState:
        if state["skip"]:
            return state
        profile = state["session"].profile
        before = asdict(profile)
        extract_slots(state["user_msg"], profile)
        state["slots_changed"] = asdict(profile) != before
        return state

    def node_cache(state: ChatState) -> ChatState:
        # Runs AFTER slot extraction: the cache key includes the updated profile, and a
        # hit can no longer swallow constraints stated in this message.
        if state["skip"]:
            return state
        session = state["session"]
        hit = session.cache.lookup(state["user_msg"], _cache_context(session))
        if hit:
            state.update(
                action="cache", response=hit.answer, category=hit.category, skip=True,
                matched_products=[p for p in (get_by_id(i) for i in hit.product_ids) if p],
            )
        return state

    def node_clarify(state: ChatState) -> ChatState:
        if state["skip"]:
            return state
        session = state["session"]
        msg = state["user_msg"]
        asked_already = session.metrics and session.metrics[-1].get("action") == "clarify"
        if (session.profile.is_empty() and not session.last_product_ids and not asked_already
                and _is_vague(msg) and not find_products_by_name(msg)
                and not detect_superlative(msg.lower()) and not is_list_request(msg.lower())):
            session.pending_clarify = True
            state.update(action="clarify", response=prompts.CLARIFY_USE_CASE, skip=True)
        return state

    def node_retrieve(state: ChatState) -> ChatState:
        if state["skip"]:
            return state
        session = state["session"]
        profile = session.profile
        user_msg = state["user_msg"]
        msg = user_msg.lower()
        named = find_products_by_name(user_msg)
        notes: list = []

        def finish_llm(products: list, template: str, product_note: str):
            state["matched_products"] = products
            state["category"] = infer_category(msg, profile.use_case, products)
            state["notes"] = notes
            state["system_prompt"] = template.format(
                profile_summary=format_profile_summary(profile),
                product_catalog=format_products_for_prompt(products, settings.max_prompt_products),
                product_note=product_note,
                notes_block=prompts.build_notes_block(notes),
            )
            state["action"] = "llm"
            return state

        # 1. Factual question about one named product: answer from the catalog.
        if len(named) == 1 and is_factual(msg):
            answer = format_factual_response(named[0], user_msg)
            if answer:
                state.update(action="factual", response=answer, matched_products=named,
                             category=infer_category("", None, named), skip=True)
                return state

        # 2. Comparison of explicitly named products: exact lookup, not semantic search.
        wants_compare = state["intent"] == Intent.COMPARISON.value or is_comparison(msg)
        if wants_compare and len(named) >= 2:
            for p in named:
                broken = violations(p, profile)
                if broken:
                    notes.append(f"{p['name']} {'; '.join(broken)}.")
            return finish_llm(named[:4], prompts.COMPARE_PROMPT_TEMPLATE, "named by the customer")

        # 3. Constraints first, then ranking.
        fr = apply_constraints(load_products(), profile)
        if fr.hard_empty:
            state.update(action="no_match", response=_no_match_response(profile),
                         matched_products=[], category=infer_category(msg, profile.use_case, []), skip=True)
            return state
        candidates = fr.products
        if fr.relaxed:
            notes.append(
                "No product meets every requirement. These could NOT be satisfied together with the "
                f"others: {'; '.join(fr.relaxed)}. The list below contains the closest alternatives."
            )

        # 3a. "Show me all ..." -> deterministic list.
        if is_list_request(msg):
            shown = sorted(candidates, key=lambda p: p["price"])[:LIST_CAP]
            note = ""
            if fr.relaxed:
                note = f"Note: nothing matched {'; '.join(fr.relaxed)}, so that requirement was dropped."
            elif len(candidates) > LIST_CAP:
                note = f"Showing the {LIST_CAP} lowest-priced of {len(candidates)} matches."
            state.update(action="factual", response=format_list_response(shown, note),
                         matched_products=shown, category=infer_category(msg, profile.use_case, shown), skip=True)
            return state

        # 3b. Superlatives are exact sorts, not similarity ("cheapest", "lightest", "best battery").
        superlative = detect_superlative(msg)
        if superlative:
            label, key, reverse = superlative
            ranked = sorted(candidates, key=key, reverse=reverse)
            product_note = f"already filtered to the customer's constraints, sorted by {label}"
        else:
            semantic_ids = semantic_rank(user_msg, [p["id"] for p in candidates], profile.use_case)
            ranked = rank_products(candidates, profile, user_msg, semantic_ids)
            product_note = "already filtered to the customer's constraints, best match first"

        # 3c. Keep follow-ups grounded: a named product, or the products from the previous
        #     answer when this message adds no new requirement ("compare these", "the second one").
        pinned = []
        for p in named:
            pinned.append(p)
            broken = violations(p, profile)
            if broken:
                notes.append(f"The customer asked about {p['name']}, which {'; '.join(broken)}.")
        if not named and not state["slots_changed"] and not superlative and session.last_product_ids:
            previous = [get_by_id(i) for i in session.last_product_ids[:PINNED_FOLLOWUP]]
            pinned = [p for p in previous if p and passes_hard(p, profile)]
        if pinned:
            ranked = pinned + [p for p in ranked if p not in pinned]

        template = prompts.COMPARE_PROMPT_TEMPLATE if (wants_compare and pinned) else prompts.SYSTEM_PROMPT_TEMPLATE
        products = ranked[:settings.max_prompt_products]
        if wants_compare and pinned:
            products = pinned[:4]
        return finish_llm(products, template, product_note)

    # ── Edges ──
    def route(next_node: str):
        return lambda state: "end" if state["skip"] else next_node

    g = StateGraph(ChatState)
    g.add_node("intent", node_intent)
    g.add_node("slots", node_slots)
    g.add_node("cache", node_cache)
    g.add_node("clarify", node_clarify)
    g.add_node("retrieve", node_retrieve)

    g.set_entry_point("intent")
    g.add_conditional_edges("intent", route("slots"), {"end": END, "slots": "slots"})
    g.add_edge("slots", "cache")
    g.add_conditional_edges("cache", route("clarify"), {"end": END, "clarify": "clarify"})
    g.add_conditional_edges("clarify", route("retrieve"), {"end": END, "retrieve": "retrieve"})
    g.add_edge("retrieve", END)
    return g.compile()
