import pytest

from agents.catalog import find_exact_mentions, find_products_by_name, load_products
from agents.filters import apply_constraints, battery_hours, gpu_rank, passes_hard, violations
from agents.product_matcher import detect_superlative, format_factual_response, rank_products
from agents.session_manager import UserProfile
from agents.slot_filler import extract_slots

REQUIRED = ["id", "name", "series", "category", "price", "display", "processor", "ram", "storage",
            "gpu", "battery", "weight_lbs", "weight_kg", "os", "ports", "use_cases", "tags", "description"]


def names(message):
    return [p["name"] for p in find_products_by_name(message)]


def profile(message):
    return extract_slots(message, UserProfile())


def test_catalog_integrity():
    products = load_products()
    assert len(products) == 65
    assert len({p["id"] for p in products}) == len(products)
    assert len({p["name"] for p in products}) == len(products)
    for p in products:
        assert all(k in p for k in REQUIRED), p["id"]
        assert 100 <= p["price"] <= 10000 and 0 < p["weight_lbs"] < 10


@pytest.mark.parametrize("message,expected", [
    ("Compare Legion 5i vs Legion 7i", ["Legion 5i Gen 10", "Legion 7i Gen 9"]),
    ("What is the price of the ThinkPad X1 Carbon Gen 13?", ["ThinkPad X1 Carbon Gen 13"]),
    ("legion pro 5i or legion slim 5i", ["Legion Pro 5i Gen 9", "Legion Slim 5i Gen 9"]),
    ("X1 Carbon Gen 12 vs X1 Carbon Aura", ["ThinkPad X1 Carbon Gen 12", "ThinkPad X1 Carbon Gen 13 Aura"]),
    ("tell me about the T14s", ["ThinkPad T14s Gen 5"]),
    ("thinkpad t14 amd vs thinkpad t14 intel", ["ThinkPad T14 Gen 5 AMD", "ThinkPad T14 Gen 5 Intel"]),
    ("is the legion 7i gen 8 better than the legion 7i", ["Legion 7i Gen 8", "Legion 7i Gen 9"]),
    ("How much does the Legion 9i Gen 9 weigh?", ["Legion 9i Gen 9"]),
    ("I need a gaming laptop", []),
    ("show me legion laptops", []),
])
def test_find_products_by_name(message, expected):
    assert names(message) == expected


def test_find_exact_mentions_prefers_longest_name():
    text = "I recommend the **ThinkPad X1 Carbon Gen 13 Aura** ($2,099)."
    assert [p["name"] for p in find_exact_mentions(text)] == ["ThinkPad X1 Carbon Gen 13 Aura"]
    both = text + " The ThinkPad X1 Carbon Gen 13 is $1899."
    assert len(find_exact_mentions(both)) == 2


def test_helpers():
    assert gpu_rank("NVIDIA RTX 4070 Ti 12GB") > gpu_rank("NVIDIA RTX 4070 8GB") > gpu_rank("NVIDIA RTX 4060 8GB")
    assert gpu_rank("Intel Arc Graphics") == 0
    assert battery_hours({"battery": "57Wh, up to 14.5hrs"}) == 14.5
    assert battery_hours({"battery": "99.9Wh"}) is None
    assert battery_hours({"battery": "55.5Wh, up to 2hrs gaming"}) == 2.0


def test_firm_budget_is_a_hard_ceiling():
    fr = apply_constraints(load_products(), profile("gaming laptop under $1000"))
    assert fr.products and all(p["price"] <= 1000 for p in fr.products)
    assert all(p["category"] == "gaming" or gpu_rank(p["gpu"]) > 0 for p in fr.products)


def test_approximate_budget_allows_ten_percent():
    fr = apply_constraints(load_products(), profile("business laptop around $1200"))
    assert max(p["price"] for p in fr.products) <= 1320
    assert any(p["price"] > 1200 for p in fr.products)


def test_processor_and_brand_rejection_are_enforced():
    # Was: the keyword route had a 'Processor rejection' comment and no code.
    fr = apply_constraints(load_products(), profile("Budget $1500, no AMD, no IdeaPad, need dedicated GPU"))
    assert fr.products
    for p in fr.products:
        assert "amd" not in p["processor"].lower()
        assert p["series"] != "IdeaPad" and p["price"] <= 1500 and gpu_rank(p["gpu"]) > 0
    fr = apply_constraints(load_products(), profile("under $600 with Windows, not Chromebook"))
    assert fr.products and all("chromebook" not in p["name"].lower() for p in fr.products)


def test_gpu_floor_and_series():
    fr = apply_constraints(load_products(), profile("Show me all Legion laptops with RTX 4070 or better"))
    assert len(fr.products) == 10 and not fr.relaxed
    assert all(p["series"] == "Legion" and gpu_rank(p["gpu"]) >= gpu_rank("rtx 4070") for p in fr.products)


def test_weight_limit_is_kept_when_any_product_meets_it():
    fr = apply_constraints(load_products(), profile("I need a laptop that weighs less than 2.5 lbs"))
    assert fr.products and all(p["weight_lbs"] <= 2.5 for p in fr.products) and not fr.relaxed


def test_unmeetable_soft_constraint_is_relaxed_and_reported():
    # Was: constraints were dropped silently when fewer than 2 products remained.
    fr = apply_constraints(load_products(), profile("gaming laptop with at least 12 hours battery under $900"))
    assert fr.products and all(p["price"] <= 900 for p in fr.products)
    assert fr.relaxed == ["12+ hours of battery"]


def test_impossible_hard_constraint_returns_nothing():
    fr = apply_constraints(load_products(), profile("gaming laptop under $200"))
    assert fr.hard_empty and fr.products == []


def test_violations_reports_what_a_product_breaks():
    legion9 = find_products_by_name("Legion 9i")[0]
    broken = violations(legion9, profile("under $1500, no Intel"))
    assert any("budget" in b for b in broken) and any("intel" in b for b in broken)
    assert not passes_hard(legion9, profile("under $1500"))


def test_superlatives_are_exact_sorts():
    products = load_products()
    label, key, reverse = detect_superlative("what is the cheapest lenovo laptop you have?")
    assert sorted(products, key=key, reverse=reverse)[0]["price"] == 299
    label, key, reverse = detect_superlative("what thinkpad has the best battery life?")
    thinkpads = [p for p in products if p["series"] == "ThinkPad"]
    assert sorted(thinkpads, key=key, reverse=reverse)[0]["id"] == "TP-X13S-G3"
    label, key, reverse = detect_superlative("the most powerful laptop you have")
    assert sorted(products, key=key, reverse=reverse)[0]["id"] == "LG-9I-G9"
    assert detect_superlative("a good laptop") is None


def test_rank_products_fuses_semantic_order_instead_of_discarding_it():
    p = profile("gaming")
    candidates = [x for x in load_products() if x["category"] == "gaming"][:6]
    semantic = [c["id"] for c in reversed(candidates)]
    lexical_only = rank_products(candidates, p, "gaming laptop")
    fused = rank_products(candidates, p, "gaming laptop", semantic_ids=semantic)
    assert {c["id"] for c in fused} == {c["id"] for c in candidates}
    assert [c["id"] for c in fused] != [c["id"] for c in lexical_only]
    # A product ranked first by both signals stays first.
    top = lexical_only[0]
    agree = [top["id"]] + [c["id"] for c in candidates if c["id"] != top["id"]]
    assert rank_products(candidates, p, "gaming laptop", semantic_ids=agree)[0]["id"] == top["id"]


def test_factual_answers_come_from_the_catalog():
    x1 = find_products_by_name("ThinkPad X1 Carbon Gen 13")[0]
    assert "$1899" in format_factual_response(x1, "What is the price of the ThinkPad X1 Carbon Gen 13?")
    assert "2.48 lbs" in format_factual_response(x1, "how heavy is it")
    assert "Thunderbolt" in format_factual_response(x1, "full specs please")
