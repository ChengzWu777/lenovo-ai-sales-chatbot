import re
from agents.session_manager import UserProfile

BUDGET_PATTERNS = [
    (r"under\s*\$?(\d+)k", lambda m: (0, int(m.group(1)) * 1000)),
    (r"under\s*\$?([\d,]+)", lambda m: (0, int(m.group(1).replace(",", "")))),
    (r"less than\s*\$?([\d,]+)", lambda m: (0, int(m.group(1).replace(",", "")))),
    (r"below\s*\$?([\d,]+)", lambda m: (0, int(m.group(1).replace(",", "")))),
    (r"no more than\s*\$?([\d,]+)", lambda m: (0, int(m.group(1).replace(",", "")))),
    (r"not (more|over|above)\s*\$?([\d,]+)", lambda m: (0, int(m.group(2).replace(",", "")))),
    (r"\$?([\d,]+)\s*-\s*\$?([\d,]+)", lambda m: (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")))),
    (r"around\s*\$?([\d,]+)", lambda m: (int(m.group(1).replace(",", "")) * 0.85, int(m.group(1).replace(",", "")) * 1.15)),
    (r"about\s*\$?([\d,]+)", lambda m: (int(m.group(1).replace(",", "")) * 0.85, int(m.group(1).replace(",", "")) * 1.15)),
    (r"budget.*?\$?([\d,]+)", lambda m: (0, int(m.group(1).replace(",", "")))),
    (r"\$(\d[\d,]*)\s*(budget|max|maximum|limit|top)", lambda m: (0, int(m.group(1).replace(",", "")))),]

FIRM_BUDGET_PATTERNS = [
    r"firm", r"strict", r"hard limit", r"no more than", r"not (more|over|above)",
    r"do not.*above", r"don't.*above", r"absolutely", r"must.*under", r"cannot exceed"
]

USE_CASE_MAP = {
    "gaming": ["gaming", "games", "fps", "esports", "game", "play games", "gamer"],
    "business": ["business", "enterprise", "work", "office", "professional", "corporate", "it managed"],
    "creative": ["creative", "video editing", "photo editing", "design", "content creation",
                 "rendering", "3d", "animation", "artist", "filmmaker"],
    "student": ["student", "school", "college", "university", "study", "homework", "class"],
    "workstation": ["cad", "solidworks", "autocad", "engineering", "workstation", "simulation",
                    "data science", "machine learning", "ai workload"],
    "travel": ["travel", "traveling", "commute", "portable", "lightweight", "on the go", "field work"],
    "smb": ["smb", "small business", "startup", "presentation", "presentations","2-in-1", "convertible", "touchscreen", "flexible work", "meeting"],
}

PROCESSOR_MAP = {
    "intel": ["intel", "core ultra", "core i7", "core i5", "core i9"],
    "amd": ["amd", "ryzen"],
    "snapdragon": ["snapdragon", "qualcomm", "arm"],
}

DISPLAY_MAP = {
    "oled": ["oled"],
    "4k": ["4k", "uhd", "3840"],
    "2k": ["2k", "1440", "qhd", "wqxga", "2560"],
    "touch": ["touch", "touchscreen", "2-in-1", "convertible", "tablet"],
}

def extract_slots(message: str, profile: UserProfile) -> UserProfile:
    msg = message.lower()

    # Remove weight mentions before budget parsing
    # Prevents "under 3 lbs" being misread as "$3 budget"
    import re as _re
    msg_no_weight = _re.sub(r'under\s*[\d.]+\s*(lbs?|pounds?|kg)', '', msg)

    # Budget - use weight-cleaned message
    for pattern, extractor in BUDGET_PATTERNS:
        match = re.search(pattern, msg_no_weight)
        if match:
            result = extractor(match)
            profile.budget_min = int(result[0])
            profile.budget_max = int(result[1])
            break

    # Firm budget
    if any(re.search(p, msg) for p in FIRM_BUDGET_PATTERNS):
        profile.budget_firm = True
        
    # Use case
    for use_case, keywords in USE_CASE_MAP.items():
        if any(kw in msg for kw in keywords):
            profile.use_case = use_case
            break

    # Processor preference / rejection
    for proc, keywords in PROCESSOR_MAP.items():
        # Brand rejection
        brand_reject_patterns = [
        (r"no ideapad|not ideapad|don't want ideapad|avoid ideapad", "ideapad"),
        (r"no legion|not legion|don't want legion|avoid legion", "legion"),
        (r"no thinkpad|not thinkpad|don't want thinkpad|avoid thinkpad", "thinkpad"),
        (r"no thinkbook|not thinkbook|don't want thinkbook|avoid thinkbook", "thinkbook"),
    ]
    for pattern, brand in brand_reject_patterns:
        if re.search(pattern, msg):
            if brand not in profile.brand_reject:
                profile.brand_reject.append(brand)
        if any(kw in msg for kw in keywords):
            negation = any(neg in msg for neg in [
                f"no {proc}", f"not {proc}", f"don't want {proc}",
                f"do not want {proc}", f"without {proc}", f"avoid {proc}",
                f"absolutely not {proc}", f"no {keywords[0]}"
            ])
            if negation:
                if proc not in profile.processor_reject:
                    profile.processor_reject.append(proc)
            else:
                profile.processor_pref = proc

    # Weight
    weight_match = re.search(r"under\s*([\d.]+)\s*(lbs?|pounds?|kg)", msg)
    if weight_match:
        profile.weight_max = float(weight_match.group(1))

    # Battery
    battery_match = re.search(r"(\d+)\s*hours?\s*(battery|life|minimum|at least)?", msg)
    if battery_match:
        hours = int(battery_match.group(1))
        if 6 <= hours <= 30:
            profile.battery_min = hours

    # Display
    for display_type, keywords in DISPLAY_MAP.items():
        if any(kw in msg for kw in keywords):
            profile.display_pref = display_type
            break

    # Form factor
    if any(kw in msg for kw in ["2-in-1", "convertible", "touch", "tablet"]):
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
        budget_str = f"Budget: up to ${profile.budget_max}"
        if profile.budget_firm:
            budget_str += " (firm)"
        parts.append(budget_str)
    if profile.weight_max:
        parts.append(f"Weight limit: {profile.weight_max} lbs")
    if profile.battery_min:
        parts.append(f"Battery: {profile.battery_min}+ hours")
    if profile.processor_pref:
        parts.append(f"Processor preference: {profile.processor_pref}")
    if profile.processor_reject:
        parts.append(f"Processor excluded: {', '.join(profile.processor_reject)}")
    if profile.display_pref:
        parts.append(f"Display: {profile.display_pref}")
    if profile.form_factor:
        parts.append(f"Form factor: {profile.form_factor}")
    return " | ".join(parts) if parts else "No constraints captured yet"