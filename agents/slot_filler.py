"""Deterministic slot extraction: turns a user message into profile constraints.

Design rules
  * Keywords match on word boundaries ("arm" must not match "warm").
  * A number is only a budget when it appears in a money context. Spec numbers
    (512GB, 16 inch, RTX 4070, 12 hours, 3 lbs, 2-in-1) are stripped first.
  * A stated budget is a ceiling (budget_firm=True). Only approximate phrasing
    ("around $1200") or "budget is flexible" makes it soft (10% tolerance).
"""
import re
from typing import Optional

from agents.catalog import find_products_by_name
from agents.session_manager import UserProfile
from agents.taxonomy import SERIES, detect_use_case
from agents.text import any_kw, has_kw, is_deprioritised, is_rejected

KG_TO_LBS = 2.20462
MIN_BUDGET = 100          # nothing in the catalog is cheaper; smaller numbers are not budgets
APPROX_LOWER = 0.8        # "around $1200" -> 960..1200 (+10% tolerance in the filter)

# ── Budget ───────────────────────────────────────────────────────────────────
_NUM = r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k\b|grand\b)?(?![a-z\d]|\.\d)"

# Numbers that are specs, not money. Removed before budget parsing.
_NON_MONEY = [
    r"\b\d+\s*-\s*in\s*-\s*\d+\b", r"\b\d+\s+in\s+1\b",                       # 2-in-1
    r"\b\d+\s*-?\s*years?\s*-?\s*old\b",                                       # 10-year-old
    r"\b(?:rtx|gtx|rx)\s*\d{3,4}(?:\s*ti)?\b",                                 # RTX 4070
    r"(?<![$\d])\b[345]0[5-9]0\b(?:\s*ti\b)?(?=\s*(?:or|and|/|\+|,|gpu|graphics|card|ti|$))",  # bare 4070
    r"\bgen(?:eration)?\s*\d+\b", r"\b\d+(?:st|nd|rd|th)\s+gen\b",
    r"\bwindows\s*\d+\b", r"\bwi-?fi\s*\d\w*\b",
    r"\b\d+(?:\.\d+)?\s*-?\s*(?:gb|tb|mb|hz|lbs?|pounds?|kgs?|kilos?|hours?|hrs?|"
    r"inch(?:es)?|wh|mah|nits|watts?|cores?|threads?|fps|ports?|usb\w*|years?|months?|days?)\b",
    r"\b\d+(?:\.\d+)?\s*\"",                                                   # 14"
    r"\b[248]k\s*(?:display|screen|monitor|resolution|oled|panel|video|gaming|editing)\b",
    r"\b20[12]\d\s*(?:model|edition|version|release)\b",
]

_CEILING = (
    r"(?:under|below|less\s+than|cheaper\s+than|no\s+more\s+than|not\s+more\s+than|"
    r"not\s+(?:over|above)|nothing\s+(?:over|above)|at\s+most|up\s+to|"
    r"max(?:imum)?(?:\s+(?:of|is|budget))?|within|cap(?:ped)?\s+(?:at|of)|"
    r"limit(?:ed)?\s+(?:to|of|is)|not\s+exceed(?:ing)?|cannot\s+exceed|can'?t\s+exceed|"
    r"(?:don'?t|do\s+not|can'?t|cannot|won'?t)\s+(?:want\s+to\s+)?(?:spend|go|pay)\s+(?:more\s+than|over|above)|"
    r"firm\s+at|stay\s+(?:under|below))"
)
_APPROX = r"(?:around|about|roughly|approximately|approx\.?|circa|~|in\s+the\s+ballpark\s+of)"
_FLOOR = r"(?:over|above|more\s+than|at\s+least|minimum(?:\s+of)?|starting\s+(?:at|from))"

_SOFT_WORDS = [r"\bflexible\b", r"\bgive or take\b", r"\bcan stretch\b", r"\bnegotiable\b",
               r"\bor so\b", r"\b\d+\s*-?\s*ish\b"]
_NO_BUDGET = [r"money is no object", r"no budget (?:limit|constraint)", r"budget (?:is )?(?:not|isn'?t) (?:an? )?(?:issue|problem|concern)",
              r"price (?:does(?:n'?t| not)|is(?:n'?t| not)) (?:matter|an? (?:issue|concern))", r"unlimited budget",
              r"don'?t care (?:about|how much)(?: the)? (?:price|cost|it costs)"]


def _money(num: str, suffix: Optional[str]) -> int:
    value = float(num.replace(",", ""))
    if suffix:
        value *= 1000
    return int(round(value))


def _strip_non_money(msg: str) -> str:
    for pattern in _NON_MONEY:
        msg = re.sub(pattern, " ", msg)
    return msg


def parse_budget(message: str) -> Optional[dict]:
    """Returns {"min", "max", "firm"} or None when the message states no budget.

    {"clear": True} means the user removed their budget ("money is no object").
    """
    raw = message.lower()
    if any(re.search(p, raw) for p in _NO_BUDGET):
        return {"clear": True}
    msg = _strip_non_money(raw)
    soft = any(re.search(p, raw) for p in _SOFT_WORDS)

    def ok(*values):
        return all(v >= MIN_BUDGET for v in values)

    # Range: "between $800 and $1200", "$800-$1200", "800 to 1200"
    for pattern in (rf"(?:between|from)\s+{_NUM}\s*(?:and|to|-|–)\s*{_NUM}",
                    rf"{_NUM}\s*(?:-|–|—|to)\s*{_NUM}"):
        m = re.search(pattern, msg)
        if m:
            lo, hi = _money(m.group(1), m.group(2)), _money(m.group(3), m.group(4))
            if m.group(4) and not m.group(2) and lo < 100:   # "1-2k"
                lo *= 1000
            if ok(lo, hi) and lo < hi:
                return {"min": lo, "max": hi, "firm": not soft}

    # Approximate: "around $1200"
    m = re.search(rf"{_APPROX}\s*{_NUM}", msg)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value):
            return {"min": int(value * APPROX_LOWER), "max": value, "firm": False}

    # Ceiling: "under $1000", "no more than 800", "up to 1.5k"
    m = re.search(rf"{_CEILING}\s*{_NUM}", msg)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value):
            return {"min": None, "max": value, "firm": not soft}

    # Trailing qualifier: "$700 max", "1000 or less", "$500 budget"
    m = re.search(rf"{_NUM}\s*(?:budget|max(?:imum)?|limit|tops|dollars|bucks|usd|or\s+(?:less|under|below|cheaper))", msg)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value):
            return {"min": None, "max": value, "firm": not soft}

    # "budget is $800", "budget of 1500", "my budget: 900"
    m = re.search(rf"budget\b[^\d$.;!?]{{0,25}}{_NUM}", msg)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value):
            return {"min": None, "max": value, "firm": not soft}

    # "I only have $300", "can spend 900"
    m = re.search(rf"(?:have|got|spend|spending|afford|pay|paying)\s+(?:only\s+|just\s+|about\s+)?{_NUM}", msg)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value) and ("$" in m.group(0) or re.search(r"spend|afford|pay", m.group(0))):
            return {"min": None, "max": value, "firm": not soft}

    # Floor only: "over $1500", "at least $1000" (requires an explicit $)
    m = re.search(rf"{_FLOOR}\s*\$\s*(\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)\s*(k\b)?", msg)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value):
            return {"min": value, "max": None, "firm": False}

    # Bare "$900", or a message that is only a number (answer to "what's your budget?")
    m = re.search(r"\$\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k\b)?(?![a-z\d])", msg)
    if not m:
        m = re.fullmatch(r"\s*(\d{1,3}(?:,\d{3})+|\d{3,5})\s*(k\b)?\s*(?:dollars|usd|bucks)?[.!]?\s*", raw)
    if m:
        value = _money(m.group(1), m.group(2))
        if ok(value):
            return {"min": None, "max": value, "firm": not soft}

    return None


# ── Other slots ──────────────────────────────────────────────────────────────
PROCESSOR_MAP = {
    "intel": ["intel", "core ultra", "core i3", "core i5", "core i7", "core i9"],
    "amd": ["amd", "ryzen"],
    "snapdragon": ["snapdragon", "qualcomm", "arm-based", "arm based", "arm processor",
                   "arm chip", "windows on arm"],
}

# Things a user can rule out besides processors.
REJECTABLE = SERIES + ["chromebook", "yoga"]

_WEIGHT_LIMIT = (
    r"(?:under|below|less\s+than|lighter\s+than|no\s+more\s+than|not\s+(?:over|above)|"
    r"nothing\s+(?:over|above)|at\s+most|max(?:imum)?(?:\s+of)?|within)"
)
_WEIGHT_UNIT = r"(lbs?|pounds?|kgs?|kilos?|kilograms?)"

_MONEY_CONTEXT = r"(?:under|below|around|about|budget|than|max|to|spend|\$)\s*\$?\s*$"


def _parse_weight(msg: str) -> Optional[float]:
    m = re.search(rf"{_WEIGHT_LIMIT}\s*(\d+(?:\.\d+)?)\s*{_WEIGHT_UNIT}\b", msg)
    if not m:
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{_WEIGHT_UNIT}\s*(?:or\s+(?:less|lighter|under|below)|max|tops)", msg)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2).startswith("k"):
        value *= KG_TO_LBS
    return round(value, 2)


def _parse_gpu(msg: str) -> Optional[str]:
    m = re.search(r"\b(?:rtx|gtx)\s*(\d{4})(\s*ti\b)?", msg)
    if not m:
        m = re.search(r"(?<![$\d])\b([345]0[5-9]0)\b(\s*ti\b)?(?=\s*(?:or|and|/|\+|,|gpu|graphics|card|ti|$))", msg)
    if m:
        return m.group(1) + (" ti" if m.group(2) else "")
    if re.search(r"\b(?:dedicated|discrete)\s+(?:\w+\s+)?(?:gpu|graphics|video card)", msg):
        return "dedicated"
    return None


def _parse_display(msg: str) -> Optional[str]:
    if has_kw(msg, "oled"):
        return "oled"
    for label in ("4k", "2k"):
        for m in re.finditer(rf"(?<![a-z0-9]){label}(?![a-z0-9])", msg):
            if not re.search(_MONEY_CONTEXT, msg[:m.start()]):   # "under 2k" is money
                return label
    if any_kw(msg, ["uhd", "3840"]):
        return "4k"
    if any_kw(msg, ["1440p", "qhd", "wqxga", "2560"]):
        return "2k"
    if any_kw(msg, ["touch", "touchscreen", "touch screen"]):
        return "touch"
    return None


def extract_slots(message: str, profile: UserProfile) -> UserProfile:
    """Update profile in place from one user message and return it."""
    msg = message.lower()

    # Budget
    budget = parse_budget(message)
    if budget and budget.get("clear"):
        profile.budget_min = profile.budget_max = None
        profile.budget_firm = False
    elif budget:
        profile.budget_min, profile.budget_max = budget["min"], budget["max"]
        profile.budget_firm = bool(budget["firm"]) and budget["max"] is not None
    elif profile.budget_max and any(re.search(p, msg) for p in _SOFT_WORDS[:4]):
        profile.budget_firm = False          # "my budget is flexible" on a later turn
    elif profile.budget_max and re.search(r"\b(?:firm|strict|hard limit|not a (?:penny|cent|dollar) more)\b", msg):
        profile.budget_firm = True

    # Use case (ignores negated mentions: "forget gaming, show me business laptops")
    previous_use_case = profile.use_case
    use_case = detect_use_case(msg)
    if use_case:
        profile.use_case = use_case

    # Series / product-line preference and rejection
    mentioned_series = []
    for term in REJECTABLE:
        plural = term + "s"
        if not (has_kw(msg, term) or has_kw(msg, plural)):
            continue
        if is_rejected(msg, term) or is_rejected(msg, plural):
            if term not in profile.brand_reject:
                profile.brand_reject.append(term)
            if term in profile.series_pref:
                profile.series_pref.remove(term)
        elif term in SERIES and not is_deprioritised(msg, term):
            mentioned_series.append(term)
            if term in profile.brand_reject:
                profile.brand_reject.remove(term)
    if mentioned_series and not find_products_by_name(message):
        # "show me ThinkPads" is a preference; "price of the ThinkPad X1 Carbon" is not.
        profile.series_pref = mentioned_series
    elif use_case and use_case != previous_use_case:
        profile.series_pref = []             # new need, old product-line preference is stale

    # Processor preference / rejection — evaluated for every vendor
    for proc, keywords in PROCESSOR_MAP.items():
        mentioned = [kw for kw in keywords if has_kw(msg, kw)]
        if not mentioned:
            continue
        if any(is_rejected(msg, kw) for kw in mentioned):
            if proc not in profile.processor_reject:
                profile.processor_reject.append(proc)
            if profile.processor_pref == proc:
                profile.processor_pref = None
        elif any(is_deprioritised(msg, kw) for kw in mentioned):
            continue                         # "Intel over AMD": no AMD preference, no hard ban
        else:
            profile.processor_pref = proc
            if proc in profile.processor_reject:
                profile.processor_reject.remove(proc)

    # Weight (kg converted to lbs)
    weight = _parse_weight(msg)
    if weight:
        profile.weight_max = weight

    # Battery
    m = re.search(r"(\d+)\+?\s*(?:hours?|hrs?)\b", msg)
    if m and 4 <= int(m.group(1)) <= 30:
        profile.battery_min = int(m.group(1))

    # GPU floor
    gpu = _parse_gpu(msg)
    if gpu:
        profile.gpu_min = gpu

    # Display
    display = _parse_display(msg)
    if display:
        profile.display_pref = display

    # Screen size
    m = re.search(r"\b(1[0-8](?:\.\d)?)\s*(?:-\s*)?(?:inch(?:es)?\b|\")", msg)
    if m:
        profile.screen_size_pref = float(m.group(1))

    # Form factor
    if any_kw(msg, ["2-in-1", "2 in 1", "convertible", "tablet mode", "detachable"]):
        profile.form_factor = "2-in-1"

    return profile


def get_missing_slots(profile: UserProfile) -> list[str]:
    missing = []
    if not profile.use_case:
        missing.append("use_case")
    if not profile.budget_max:
        missing.append("budget")
    return missing


def format_profile_summary(profile: UserProfile) -> str:
    parts = []
    if profile.use_case:
        parts.append(f"Use case: {profile.use_case}")
    if profile.budget_max:
        kind = "hard limit" if profile.budget_firm else "approximate"
        parts.append(f"Budget: up to ${profile.budget_max} ({kind})")
    elif profile.budget_min:
        parts.append(f"Budget: at least ${profile.budget_min}")
    if profile.series_pref:
        parts.append(f"Preferred line: {', '.join(profile.series_pref)}")
    if profile.brand_reject:
        parts.append(f"Excluded: {', '.join(profile.brand_reject)}")
    if profile.weight_max:
        parts.append(f"Weight limit: {profile.weight_max} lbs")
    if profile.battery_min:
        parts.append(f"Battery: {profile.battery_min}+ hours")
    if profile.processor_pref:
        parts.append(f"Processor preference: {profile.processor_pref}")
    if profile.processor_reject:
        parts.append(f"Processor excluded: {', '.join(profile.processor_reject)}")
    if profile.gpu_min:
        gpu = "dedicated GPU" if profile.gpu_min == "dedicated" else f"RTX {profile.gpu_min.upper()} or better"
        parts.append(f"GPU: {gpu}")
    if profile.display_pref:
        parts.append(f"Display: {profile.display_pref}")
    if profile.screen_size_pref:
        parts.append(f"Screen size: ~{profile.screen_size_pref:g}\"")
    if profile.form_factor:
        parts.append(f"Form factor: {profile.form_factor}")
    return " | ".join(parts) if parts else "No constraints captured yet"
