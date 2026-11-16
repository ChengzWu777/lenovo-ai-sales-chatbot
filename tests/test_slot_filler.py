import pytest

from agents.session_manager import UserProfile
from agents.slot_filler import extract_slots, parse_budget
from evaluation.test_cases import TEST_CASES, TEST_CASES_V2


def slots(message, profile=None):
    return extract_slots(message, profile or UserProfile())


# ── Regressions for bugs found in review ─────────────────────────────────────

@pytest.mark.parametrize("message", [
    "I want a gaming laptop but I absolutely do NOT want AMD processor",
    "gaming laptop, no AMD",
    "anything but AMD",
    "I don't want a Ryzen",
])
def test_processor_rejection_is_recorded(message):
    # Was: mis-indented loop, only 'snapdragon' was ever evaluated.
    p = slots(message)
    assert p.processor_reject == ["amd"]
    assert p.processor_pref is None


def test_processor_preference_is_recorded():
    assert slots("I prefer Intel").processor_pref == "intel"
    assert slots("Ryzen would be great").processor_pref == "amd"


def test_intel_over_amd_prefers_intel_without_banning_amd():
    p = slots("I prefer Intel over AMD, need a business laptop around $1200")
    assert p.processor_pref == "intel"
    assert p.processor_reject == []


def test_arm_does_not_match_inside_other_words():
    # Was: 'warm' / 'pharmacy' -> processor_pref = snapdragon
    p = slots("something to keep me warm, just a pharmacy student")
    assert p.processor_pref is None


def test_later_message_can_lift_a_rejection():
    p = slots("no AMD")
    slots("actually AMD is fine", p)
    assert p.processor_reject == [] and p.processor_pref == "amd"


@pytest.mark.parametrize("message", [
    "budget laptop with 512GB SSD",        # was budget_max=512
    "budget laptop with 16GB RAM",         # was budget_max=16
    "I want a 2-in-1",                     # was parsed as the range 2-1
    "must be under 3 lbs",
    "RTX 4070 or better, budget flexible",
    "a 16 inch screen with 144hz",
    "ThinkPad X1 Carbon Gen 13",
    "something for my 10-year-old",
    "battery over 12 hours",
    "Windows 11 laptop for under 4 lbs",
])
def test_spec_numbers_are_not_budgets(message):
    assert parse_budget(message) is None
    assert slots(message).budget_max is None


@pytest.mark.parametrize("message,lo,hi,firm", [
    ("I need a gaming laptop under $1000", None, 1000, True),
    ("under 1.5k", None, 1500, True),
    ("between $800 and $1200", 800, 1200, True),
    ("$1,200 - $1,500 for video editing", 1200, 1500, True),
    ("800 to 1200", 800, 1200, True),
    ("no more than 900", None, 900, True),
    ("my budget is $800, no more than that", None, 800, True),
    ("Student budget $500", None, 500, True),
    ("$700 max", None, 700, True),
    ("I only have $300, what can I get?", None, 300, True),
    ("I can spend up to 1500", None, 1500, True),
    ("I have 2 kids and 900 dollars", None, 900, True),
    ("800", None, 800, True),
    ("around $1200", 960, 1200, False),
    ("about 1.5k", 1200, 1500, False),
    ("under $1000 but I'm flexible", None, 1000, False),
    ("under 2k with a 4k display", None, 2000, True),
])
def test_budget_parsing(message, lo, hi, firm):
    p = slots(message)
    assert (p.budget_min, p.budget_max, p.budget_firm) == (lo, hi, firm)


def test_budget_can_be_removed():
    p = slots("under $1000")
    slots("actually money is no object", p)
    assert p.budget_max is None and p.budget_firm is False


def test_budget_updates_on_later_turn():
    p = slots("gaming laptop under $1000")
    slots("ok I can go up to 1500", p)
    assert p.budget_max == 1500 and p.use_case == "gaming"


@pytest.mark.parametrize("message,expected", [
    ("under 3 lbs", 3.0),
    ("nothing over 4 lbs", 4.0),          # was not parsed
    ("weighs less than 2.5 lbs", 2.5),
    ("under 1.5 kg", 3.31),               # was stored as 1.5 lbs
    ("5 pounds or less", 5.0),
])
def test_weight_parsing(message, expected):
    assert slots(message).weight_max == expected


def test_weight_limit_is_not_a_budget_and_not_firm():
    p = slots("I want a ThinkPad for business travel, must be under 3 lbs")
    assert p.budget_max is None and p.budget_firm is False and p.weight_max == 3.0


@pytest.mark.parametrize("message,use_case", [
    ("laptop for video editing work", "creative"),       # generic 'work' must not win
    ("I need a laptop for architecture school, CAD and rendering", "workstation"),
    ("actually forget gaming, show me business laptops", "business"),
    ("Student budget $500, mainly for school work and some light gaming", "student"),
    ("my program keeps crashing", None),                  # 'ram' inside 'program'
])
def test_use_case(message, use_case):
    assert slots(message).use_case == use_case


def test_series_preference_and_rejection():
    assert slots("show me ThinkPads").series_pref == ["thinkpad"]
    p = slots("Budget $1500, no AMD, no IdeaPad, need dedicated GPU")
    assert p.brand_reject == ["ideapad"] and p.processor_reject == ["amd"]
    assert p.gpu_min == "dedicated" and p.series_pref == []
    assert slots("under $600 with Windows, not Chromebook").brand_reject == ["chromebook"]


def test_named_product_is_not_a_series_preference():
    assert slots("What is the price of the ThinkPad X1 Carbon Gen 13?").series_pref == []


def test_stale_series_preference_is_dropped_when_need_changes():
    p = slots("show me ThinkPads for business")
    slots("what about something for gaming?", p)
    assert p.series_pref == [] and p.use_case == "gaming"


def test_gpu_display_battery_screen_form_factor():
    p = slots("Legion with RTX 4070 Ti or better, 16 inch OLED, at least 8 hours battery")
    assert (p.gpu_min, p.display_pref, p.screen_size_pref, p.battery_min) == ("4070 ti", "oled", 16.0, 8)
    assert slots("a convertible please").form_factor == "2-in-1"
    assert slots("under 2k").display_pref is None          # '2k' here is money


# ── The labels the old evaluator injected are now assertions on extraction ────

@pytest.mark.parametrize("case", TEST_CASES + TEST_CASES_V2, ids=lambda c: f"case{c['id']}")
def test_eval_case_labels_are_extracted_from_the_query(case):
    p = slots(case["query"])
    assert p.budget_max == case.get("budget_max")
    assert p.budget_firm == bool(case.get("budget_firm"))
    assert p.weight_max == case.get("weight_max")
