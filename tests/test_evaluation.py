"""The evaluator itself is code that can be wrong — these tests pin its checks."""
import asyncio

from agents.catalog import find_products_by_name
from agents.sales_graph import build_sales_graph
from evaluation import checks
from evaluation.evaluator import EchoLLM, Evaluator, normalize_case, summarize
from evaluation.test_cases import MULTI_TURN_CASES, TEST_CASES, TEST_CASES_V2
from tests.conftest import FakeLLM


def product(name):
    return find_products_by_name(name)[0]


def test_extract_prices_handles_commas():
    # Was: r"\$(\d{3,4})" missed "$1,299", so an over-budget answer could pass.
    assert checks.extract_prices("The X is $1,299, down from $1499.00; save $200") == [1299, 1499, 200]


def test_check_keywords():
    assert checks.check_keywords("A ThinkPad with OLED", any_of=["legion", "thinkpad"])["ok"]
    assert not checks.check_keywords("A ThinkPad", any_of=["legion"])["ok"]
    assert not checks.check_keywords("A ThinkPad", all_of=["thinkpad", "oled"])["ok"]
    assert checks.check_keywords("AMD Ryzen 7", none_of=["ryzen 7"])["found_forbidden"] == ["ryzen 7"]


def test_check_constraints_uses_catalog_facts():
    legion9 = product("Legion 9i")
    out = checks.check_constraints([legion9], max_price=1000, reject_processor=["intel"], reject_terms=["legion"])
    assert not out["ok"] and len(out["problems"]) == 3
    assert checks.check_constraints([legion9], max_price=4000)["ok"]


def test_check_grounding_flags_products_not_provided():
    provided = [product("Legion 5i Gen 9")]
    ok = checks.check_grounding("I recommend the **Legion 5i Gen 9** at $1,399.", provided)
    assert ok["ok"] and ok["mentioned"] == ["Legion 5i Gen 9"]
    bad = checks.check_grounding("Get the Legion 9i Gen 9 instead.", provided)
    assert bad["ungrounded"] == ["Legion 9i Gen 9"]


def test_check_grounding_flags_wrong_price():
    provided = [product("Legion 5i Gen 9")]
    bad = checks.check_grounding("The Legion 5i Gen 9 ($1,299) is great.", provided)
    assert not bad["ok"] and "catalog $1399" in bad["wrong_prices"][0]


def test_check_grounding_does_not_confuse_sibling_models():
    # Found on the first real-LLM run: "ThinkBook 14 Gen 7 Intel ($799)" was reported as
    # "ThinkBook 14 Gen 7: said $799, catalog $749" because one name is a prefix of the other.
    provided = [product("ThinkBook 14 Gen 7"), product("ThinkBook 14 Gen 7 Intel"),
                product("ThinkPad X1 Carbon Gen 13"), product("X1 Carbon Aura")]
    text = ("- **ThinkBook 14 Gen 7** ($749) is the value pick.\n"
            "- **ThinkBook 14 Gen 7 Intel** ($799) adds Arc graphics.\n"
            "- **ThinkPad X1 Carbon Gen 13 Aura** ($2,099) has the best webcam; "
            "the ThinkPad X1 Carbon Gen 13 is $1,899.")
    out = checks.check_grounding(text, provided)
    assert out["ok"], out
    wrong = checks.check_grounding("The **ThinkBook 14 Gen 7 Intel** ($749) is great.", provided)
    assert wrong["wrong_prices"] == ["ThinkBook 14 Gen 7 Intel: said $749, catalog $799"]


def test_budget_ceiling():
    assert checks.budget_ceiling(1000, True) == 1000
    assert round(checks.budget_ceiling(1000, False)) == 1100
    assert checks.budget_ceiling(None, True) is None


def test_single_turn_case_is_normalised_without_touching_the_profile():
    case = normalize_case(TEST_CASES[0])
    assert case["turns"][0]["user"] == TEST_CASES[0]["query"]
    expect = case["turns"][0]["expect"]
    assert expect["profile"] == {"budget_max": 1000, "budget_firm": True, "weight_max": None}
    assert expect["max_price"] == 1000 and expect["category"] == "gaming"


def test_case_ids_are_unique_and_suites_have_expected_size():
    all_cases = TEST_CASES + TEST_CASES_V2 + MULTI_TURN_CASES
    assert (len(TEST_CASES), len(TEST_CASES_V2)) == (20, 20) and len(MULTI_TURN_CASES) >= 10
    assert len({c["id"] for c in all_cases}) == len(all_cases)


def run(cases, llm, mode):
    graph = build_sales_graph(retriever=None)
    return asyncio.run(Evaluator(graph, llm, mode).run_all(cases, verbose=False))


def test_evaluator_does_not_inject_labels():
    # A case whose labels say "$500 firm" but whose query states no budget must FAIL
    # the slots check instead of silently getting a $500 profile (the old behaviour).
    case = {"id": 999, "description": "label without evidence", "query": "I need a gaming laptop",
            "expected_category": "gaming", "budget_max": 500, "budget_firm": True, "must_include": []}
    result = run([case], EchoLLM(), "stub")[0]
    assert result.failed("slots") and result.failed("constraints")
    assert any(p > 500 for p in checks.extract_prices(result.turns[0].response))


def test_evaluator_catches_an_ungrounded_answer():
    case = {"id": 998, "description": "hallucination", "query": "I need a gaming laptop under $1000",
            "expected_category": "gaming", "budget_max": 1000, "budget_firm": True, "must_include": []}
    liar = FakeLLM(answer="You should buy the Legion 9i Gen 9 for $999, it is the best gaming laptop there is.")
    result = run([case], liar, "real")[0]
    assert result.failed("grounding") and result.failed("constraints")
    assert result.score() < 0.6


def test_multi_turn_suite_passes_on_the_deterministic_pipeline():
    results = run(MULTI_TURN_CASES, EchoLLM(), "stub")
    failures = {r.case_id: [t.failures for t in r.turns if t.failures] for r in results if not r.passed}
    assert failures == {}
    summary = summarize(results)
    assert summary["cases"] == len(MULTI_TURN_CASES) and summary["constraints_pass_pct"] == 100.0
    assert summary["actions"]["cache"] == 1 and summary["actions"]["no_match"] == 1


def test_single_turn_suites_never_violate_constraints_or_slots():
    results = run(TEST_CASES + TEST_CASES_V2, EchoLLM(), "stub")
    assert [r.case_id for r in results if r.failed("constraints")] == []
    assert [r.case_id for r in results if r.failed("slots")] == []
