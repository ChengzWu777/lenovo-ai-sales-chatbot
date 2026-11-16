"""End-to-end evaluation.

Every case runs through chat_service.run_turn — the same function the WebSocket
handler calls — with an empty profile, the production temperature and the real
conversation history. Nothing from the labels is fed into the pipeline.

Two LLM modes
  real  ChatOllama, as configured in config.py
  stub  EchoLLM: answers with the product list it was given. No GPU needed. It
        cannot measure answer quality, but it exercises routing, slot extraction,
        filtering, retrieval, caching and history, and it answers the question
        "was the right product in front of the model?"

Dimensions (each pass/fail per case)
  slots        profile extracted from the conversation matches the labels
  routing      action / category / LLM-called match what the case expects
  constraints  every product shown to the LLM and every product named in the
               answer respects budget and exclusions (checked against the catalog)
  grounding    products named in the answer were provided; prices quoted match
  keywords     must_include / must_not_include on the answer text
"""
import statistics
import time
from dataclasses import dataclass, field

from agents.catalog import find_exact_mentions
from agents.session_manager import Session
from chat_service import run_turn
from evaluation import checks

WEIGHTS = {"constraints": 0.30, "grounding": 0.20, "keywords": 0.20, "slots": 0.15, "routing": 0.15}
DIMENSIONS = list(WEIGHTS)


class EchoLLM:
    """Deterministic stand-in: replies with the product lines from the system prompt."""

    async def astream(self, messages):
        system = messages[0].content
        lines = [ln for ln in system.splitlines() if ln.startswith("- ") and " | $" in ln]
        text = "Products considered for this answer:\n" + "\n".join(lines)
        for line in text.splitlines(keepends=True):
            yield type("Chunk", (), {"content": line, "usage_metadata": None})()


@dataclass
class TurnRecord:
    user: str
    action: str
    category: str
    response: str
    product_ids: list
    latency_s: float
    ttft_s: float | None
    llm_called: bool
    fallback: bool
    failures: dict = field(default_factory=dict)   # dimension -> list of reasons


@dataclass
class CaseResult:
    case_id: int
    description: str
    turns: list = field(default_factory=list)

    def failed(self, dimension: str) -> bool:
        return any(dimension in t.failures for t in self.turns)

    def score(self) -> float:
        return round(sum(w for d, w in WEIGHTS.items() if not self.failed(d)), 2)

    @property
    def passed(self) -> bool:
        return not any(t.failures for t in self.turns)

    def to_dict(self) -> dict:
        return {
            "id": self.case_id, "description": self.description, "score": self.score(),
            "passed": self.passed,
            **{f"{d}_ok": not self.failed(d) for d in DIMENSIONS},
            "turns": [{
                "user": t.user, "action": t.action, "category": t.category,
                "llm_called": t.llm_called, "fallback": t.fallback,
                "latency_s": t.latency_s, "ttft_s": t.ttft_s,
                "product_ids": t.product_ids, "failures": t.failures, "response": t.response,
            } for t in self.turns],
        }


def normalize_case(case: dict) -> dict:
    """Turn a single-turn case into a one-turn conversation with an 'expect' block."""
    if "turns" in case:
        return case
    expect = {
        "profile": {
            "budget_max": case.get("budget_max"),
            "budget_firm": bool(case.get("budget_firm")),
            "weight_max": case.get("weight_max"),
        },
        "max_price": checks.budget_ceiling(case.get("budget_max"), bool(case.get("budget_firm"))),
        "must_include": case.get("must_include", []),
        "must_not_include": case.get("must_not_include", []),
        "reject_processor": case.get("reject_processor", []),
        "reject_terms": case.get("reject_terms", []),
    }
    if case.get("expected_category", "all") != "all":
        expect["category"] = case["expected_category"]
    return {"id": case["id"], "description": case["description"],
            "turns": [{"user": case["query"], "expect": expect}]}


class Evaluator:
    def __init__(self, graph, llm, llm_mode: str = "real", cfg=None):
        self.graph, self.llm, self.llm_mode = graph, llm, llm_mode
        self.cfg = cfg

    async def run_case(self, case: dict) -> CaseResult:
        case = normalize_case(case)
        result = CaseResult(case["id"], case["description"])
        session = Session(session_id=f"eval_{case['id']}")
        previous_ids: list = []

        for turn in case["turns"]:
            chunks: list = []

            async def emit(text: str):
                chunks.append(text)

            kwargs = {"cfg": self.cfg} if self.cfg else {}
            tr = await run_turn(session, turn["user"], self.graph, self.llm, emit, **kwargs)
            record = TurnRecord(
                user=turn["user"], action=tr.action, category=tr.category, response=tr.response,
                product_ids=[p["id"] for p in tr.products], latency_s=tr.latency_s, ttft_s=tr.ttft_s,
                llm_called=tr.llm_called, fallback=tr.fallback,
            )
            self._check_turn(record, tr, session, turn.get("expect", {}), previous_ids)
            result.turns.append(record)
            if tr.products:
                previous_ids = [p["id"] for p in tr.products]
        return result

    def _check_turn(self, record, tr, session, expect: dict, previous_ids: list):
        def fail(dimension: str, reason: str):
            record.failures.setdefault(dimension, []).append(reason)

        # slots
        if expect.get("profile"):
            out = checks.check_profile(session.profile, expect["profile"])
            for key, diff in out["wrong"].items():
                fail("slots", f"{key}: expected {diff['expected']!r}, got {diff['actual']!r}")

        # routing
        if "action" in expect and tr.action != expect["action"]:
            fail("routing", f"action: expected {expect['action']}, got {tr.action}")
        if "category" in expect and tr.category != expect["category"]:
            fail("routing", f"category: expected {expect['category']}, got {tr.category}")
        if "llm_called" in expect and tr.llm_called != expect["llm_called"]:
            fail("routing", f"llm_called: expected {expect['llm_called']}, got {tr.llm_called}")
        if tr.fallback:
            fail("routing", "LLM call failed or timed out; catalog fallback was sent")
        if expect.get("relaxed") and not tr.notes:
            fail("routing", "expected a relaxed-constraint note, none was produced")
        if expect.get("products_subset_of_previous") and not set(record.product_ids) <= set(previous_ids):
            fail("routing", "products are not a subset of the previous turn's products")
        if "product_ids" in expect and record.product_ids != expect["product_ids"]:
            fail("routing", f"products: expected {expect['product_ids']}, got {record.product_ids}")

        # constraints — products shown to the model AND products the answer names
        mentioned = find_exact_mentions(tr.response) if tr.llm_called else []
        to_check = {p["id"]: p for p in list(tr.products) + mentioned}.values()
        if tr.action != "no_match":
            out = checks.check_constraints(
                to_check, max_price=expect.get("max_price"),
                reject_processor=expect.get("reject_processor", ()),
                reject_terms=expect.get("reject_terms", ()),
            )
            for problem in out["problems"]:
                fail("constraints", problem)

        # grounding — only meaningful for generated text
        if tr.llm_called and not tr.fallback and self.llm_mode == "real":
            out = checks.check_grounding(tr.response, tr.products)
            for name in out["ungrounded"]:
                fail("grounding", f"recommends {name}, which was not in the provided list")
            for wrong in out["wrong_prices"]:
                fail("grounding", f"wrong price — {wrong}")

        # keywords
        out = checks.check_keywords(
            tr.response, any_of=expect.get("must_include", ()),
            all_of=expect.get("must_include_all", ()), none_of=expect.get("must_not_include", ()),
        )
        if out["missing_any"]:
            fail("keywords", f"none of {out['missing_any']} found")
        if out["missing_all"]:
            fail("keywords", f"missing {out['missing_all']}")
        if out["found_forbidden"]:
            fail("keywords", f"forbidden {out['found_forbidden']} found")

    async def run_all(self, cases: list, verbose: bool = True) -> list:
        results = []
        for i, case in enumerate(cases):
            started = time.monotonic()
            result = await self.run_case(case)
            results.append(result)
            if verbose:
                status = "PASS" if result.passed else "FAIL"
                print(f"[{i + 1:02d}/{len(cases)}] {status} #{result.case_id} {result.description} "
                      f"(score {result.score():.2f}, {time.monotonic() - started:.1f}s)")
                for t in result.turns:
                    for dimension, reasons in t.failures.items():
                        for reason in reasons:
                            print(f"         - {dimension}: {reason}")
        return results


def summarize(results: list) -> dict:
    n = len(results)
    turns = [t for r in results for t in r.turns]
    llm_turns = [t for t in turns if t.llm_called]
    latencies = sorted(t.latency_s for t in llm_turns)
    ttfts = [t.ttft_s for t in llm_turns if t.ttft_s is not None]
    actions: dict = {}
    for t in turns:
        actions[t.action] = actions.get(t.action, 0) + 1

    def pct(k):
        return round(100 * k / n, 1) if n else 0.0

    summary = {
        "cases": n, "turns": len(turns),
        "cases_passed": sum(r.passed for r in results), "pass_rate_pct": pct(sum(r.passed for r in results)),
        "avg_score": round(sum(r.score() for r in results) / n, 3) if n else 0.0,
        "actions": actions,
        "llm_call_rate_pct": round(100 * len(llm_turns) / len(turns), 1) if turns else 0.0,
        "fallbacks": sum(t.fallback for t in turns),
    }
    for d in DIMENSIONS:
        ok = sum(not r.failed(d) for r in results)
        summary[f"{d}_pass"] = f"{ok}/{n}"
        summary[f"{d}_pass_pct"] = pct(ok)
    if latencies:
        summary["llm_latency_avg_s"] = round(statistics.mean(latencies), 2)
        summary["llm_latency_p50_s"] = round(statistics.median(latencies), 2)
        summary["llm_latency_p95_s"] = round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 2)
    if ttfts:
        summary["ttft_avg_s"] = round(statistics.mean(ttfts), 2)
    return summary
