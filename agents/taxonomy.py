"""Single source of truth for use-case / category vocabulary.

Previously three separate keyword tables (slot filler, keyword router, graph node)
drifted apart. Every module now imports from here.
"""
from typing import Optional

from agents.text import has_kw, is_deprioritised, is_rejected

# (keyword, weight). Generic words get 0.5 so that a specific signal wins:
# "video editing work" is creative, not business.
# Dict order is the tie-break order: most specific use case first.
USE_CASE_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "workstation": [
        ("cad", 1), ("solidworks", 1), ("autocad", 1), ("catia", 1), ("revit", 1),
        ("engineering", 1), ("workstation", 1), ("simulation", 1), ("isv", 1),
        ("data science", 1), ("machine learning", 1), ("deep learning", 1),
        ("ai development", 1), ("ai workload", 1), ("professional gpu", 1),
    ],
    "creative": [
        ("creative", 1), ("video editing", 1), ("photo editing", 1), ("content creation", 1),
        ("rendering", 1), ("3d", 1), ("animation", 1), ("artist", 1), ("filmmaker", 1),
        ("photographer", 1), ("illustrator", 1), ("color grading", 1), ("color accurate", 1),
        ("color accuracy", 1), ("blender", 1), ("premiere", 1), ("photoshop", 1),
        ("design", 0.5), ("oled", 0.5),
    ],
    "gaming": [
        ("gaming", 1), ("games", 1), ("game", 1), ("gamer", 1), ("fps", 1), ("esports", 1),
        ("twitch", 1), ("streaming", 1), ("high refresh", 1), ("144hz", 1), ("165hz", 1),
        ("240hz", 1), ("competitive", 0.5),
    ],
    "smb": [
        ("smb", 1), ("small business", 1), ("startup", 1), ("presentation", 1),
        ("presentations", 1), ("2-in-1", 1), ("convertible", 1), ("touchscreen", 1),
    ],
    "student": [
        ("student", 1), ("students", 1), ("school", 1), ("college", 1), ("university", 1),
        ("study", 1), ("homework", 1), ("class", 1), ("classes", 1), ("education", 1),
        ("year-old", 1), ("kid", 1), ("kids", 1), ("child", 1),
    ],
    "travel": [
        ("travel", 1), ("traveling", 1), ("travelling", 1), ("commute", 1), ("portable", 1),
        ("lightweight", 1), ("ultralight", 1), ("on the go", 1), ("field work", 1),
        ("frequent flyer", 1), ("frequent flyers", 1), ("coffee shop", 1),
    ],
    "business": [
        ("business", 1), ("enterprise", 1), ("office", 1), ("corporate", 1), ("it managed", 1),
        ("executive", 1), ("vpro", 1), ("remote work", 1),
        ("professional", 0.5), ("work", 0.5),
    ],
}

# Use case -> category used for CTA / cross-sell and reporting.
USE_CASE_TO_CATEGORY = {
    "workstation": "workstation", "creative": "creative", "gaming": "gaming",
    "smb": "smb", "student": "student", "travel": "business", "business": "business",
}

SERIES = ["thinkpad", "thinkbook", "legion", "ideapad"]
SERIES_TO_CATEGORY = {"legion": "gaming", "thinkpad": "business", "thinkbook": "smb", "ideapad": "all"}


def detect_use_case(msg: str) -> Optional[str]:
    """Best use case mentioned in this message, ignoring negated mentions. msg is lowercase."""
    best, best_score = None, 0.0
    for use_case, keywords in USE_CASE_KEYWORDS.items():
        score = 0.0
        for kw, weight in keywords:
            if has_kw(msg, kw) and not is_rejected(msg, kw) and not is_deprioritised(msg, kw):
                score += weight
        if score > best_score:  # strict '>' keeps dict order as tie-break
            best, best_score = use_case, score
    return best


def infer_category(msg: str, use_case: Optional[str], products: list) -> str:
    """Category for CTA/metrics: this message, then the profile, then the top product."""
    current = detect_use_case(msg.lower())
    if current:
        return USE_CASE_TO_CATEGORY[current]
    if use_case:
        return USE_CASE_TO_CATEGORY.get(use_case, "all")
    if products:
        top = products[0]
        if top.get("category") in ("gaming", "workstation"):
            return top["category"]
        return SERIES_TO_CATEGORY.get(top.get("series", "").lower(), "all")
    return "all"
