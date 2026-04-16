import json
import os
import re
from agents.session_manager import UserProfile

PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")

CATEGORY_KEYWORDS = {
    "gaming": [
        "gaming", "game", "fps", "esports", "gamer", "legion", "rtx",
        "streaming", "twitch", "competitive", "4070", "4080", "4090",
        "high refresh", "144hz", "165hz", "240hz", "overclocking"
    ],
    "business": [
        "business", "enterprise", "thinkpad", "work", "office", "corporate",
        "it managed", "professional", "executive", "travel", "lightweight",
        "vpro", "amd pro", "fingerprint", "ir camera", "tpm", "manageable"
    ],
    "smb": [
    "thinkbook", "smb", "small business", "startup", "mid-size",
    "affordable business", "modern design", "2-in-1", "convertible",
    "presentation", "touchscreen", "tablet mode", "flexible"
    ],
    "creative": [
        "creative", "video editing", "photo editing", "design", "oled",
        "color accuracy", "rendering", "3d", "animation", "content creation",
        "filmmaker", "photographer", "illustrator", "color grading"
    ],
    "workstation": [
        "cad", "solidworks", "autocad", "catia", "revit", "engineering",
        "workstation", "isv", "simulation", "data science", "machine learning",
        "professional gpu", "quadro", "rtx ada", "p1", "p16"
    ],
    "student": [
        "student", "school", "college", "university", "study", "homework",
        "class", "budget laptop", "first laptop", "chromebook", "education"
    ],
    "budget": [
        "cheap", "cheapest", "affordable", "under $500", "under $400",
        "under $300", "entry level", "basic", "value", "budget"
    ],
}

def load_products() -> list:
    with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["products"]

def route_query(message: str, profile: UserProfile) -> tuple[str, list]:
    msg = message.lower()
    all_products = load_products()

    # Cross-category queries — bypass category routing
    cross_category_patterns = [
    r"\bcheapest\b", r"most affordable", r"lowest price",
    r"least expensive", r"what.*cheapest",
    r"^show me all$", r"^list all$",
    ]
    if any(re.search(p, msg) for p in cross_category_patterns):
        filtered = apply_profile_filters(all_products, profile)
        filtered.sort(key=lambda p: p["price"])
        return "all", filtered[:15]

    # Weight-only queries — search across all series
    weight_only = re.search(r"under\s*([\d.]+)\s*(lbs?|pounds?|kg)", msg)
    if weight_only and not any(kw in msg for kw in ["gaming", "thinkpad", "legion", "business"]):
        filtered = apply_profile_filters(all_products, profile)
        return "all", filtered[:15]

    # Score each category
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in msg:
                scores[cat] += 1

    # Boost from profile
    if profile.use_case:
        use_case = profile.use_case.lower()
        for cat in CATEGORY_KEYWORDS:
            if use_case in cat or cat in use_case:
                scores[cat] += 3

    # Pick winning category
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] == 0:
        best_cat = "all"

    # Filter products by category
    filtered = filter_by_category(all_products, best_cat, profile)

    # If too few results, expand to all
    if len(filtered) < 3:
        filtered = filter_by_profile_only(all_products, profile)
        best_cat = "all"
        # GPU filter from message
    gpu_filter = None
    if "4090" in msg:
        gpu_filter = ["rtx 4090"]
    elif "4080" in msg:
        gpu_filter = ["rtx 4080", "rtx 4090"]
    elif "4070" in msg:
        gpu_filter = ["rtx 4070", "rtx 4070 ti", "rtx 4080", "rtx 4090"]

    if gpu_filter:
        filtered = [p for p in filtered if any(
            g in p.get("gpu", "").lower() for g in gpu_filter
        )]
    return best_cat, filtered


def filter_by_category(products: list, category: str, profile: UserProfile) -> list:
    if category == "all":
        return filter_by_profile_only(products, profile)

    category_map = {
        "gaming":      lambda p: p["category"] in ["gaming"] or p["series"] == "Legion",
        "business":    lambda p: p["series"] == "ThinkPad",
        "smb":         lambda p: p["series"] == "ThinkBook" or any(t in p.get("tags", []) for t in ["2-in-1", "convertible", "touchscreen"]),
        "creative":    lambda p: any(t in p.get("tags", []) for t in
                                     ["OLED", "creator laptop", "content creation", "creative SMB"])
                                 or "oled" in str(p.get("display", "")).lower(),
        "workstation": lambda p: p["category"] == "workstation",
        "student":     lambda p: any(t in p.get("tags", []) for t in
                                     ["student", "budget", "entry level", "education", "chromebook"]),
        "budget":      lambda p: p["price"] <= 600,
    }

    filter_fn = category_map.get(category, lambda p: True)
    filtered = [p for p in products if filter_fn(p)]
    filtered = apply_profile_filters(filtered, profile)

    if len(filtered) < 3:
        filtered = apply_profile_filters(products, profile)

    return filtered


def filter_by_profile_only(products: list, profile: UserProfile) -> list:
    return apply_profile_filters(products, profile)


def apply_profile_filters(products: list, profile: UserProfile) -> list:
    result = list(products)

    # Budget filter
    if profile.budget_max:
        if profile.budget_firm:
            result = [p for p in result if p["price"] <= profile.budget_max]
        else:
            result = [p for p in result if p["price"] <= profile.budget_max * 1.1]

    # Weight filter — only apply if enough products remain
    if profile.weight_max:
        weight_filtered = [p for p in result if p.get("weight_lbs", 99) <= profile.weight_max]
        if len(weight_filtered) >= 2:
            result = weight_filtered

    # Processor rejection
    # Brand rejection
    for rejected_brand in profile.brand_reject:
        result = [p for p in result if rejected_brand.lower() not in
                  p.get("series", "").lower() and
                  rejected_brand.lower() not in p.get("name", "").lower()]

    # Processor preference (soft sort)
    if profile.processor_pref:
        pref = profile.processor_pref.lower()
        preferred = [p for p in result if pref in p.get("processor", "").lower()]
        others = [p for p in result if pref not in p.get("processor", "").lower()]
        result = preferred + others

    # Battery filter
    if profile.battery_min:
        def has_enough_battery(p):
            match = re.search(r"up to (\d+)", p.get("battery", ""))
            if match:
                return int(match.group(1)) >= profile.battery_min
            return True
        battery_filtered = [p for p in result if has_enough_battery(p)]
        if len(battery_filtered) >= 2:
            result = battery_filtered

    # Display preference
    if profile.display_pref == "oled":
        oled_products = [p for p in result if "oled" in p.get("display", "").lower()]
        if oled_products:
            result = oled_products

    # Form factor
    if profile.form_factor == "2-in-1":
        convertibles = [p for p in result if any(t in p.get("tags", [])
                        for t in ["2-in-1", "convertible", "touchscreen"])]
        if convertibles:
            result = convertibles

    return result


def format_products_for_prompt(products: list, max_products: int = 15) -> str:
    products = products[:max_products]
    lines = []
    for p in products:
        lines.append(
            f"- {p['name']} | ${p['price']} | {p['processor']} | "
            f"{p['gpu']} | {p['ram']} | {p['display']} | "
            f"{p['weight_lbs']}lbs | Battery: {p['battery']} | "
            f"Tags: {', '.join(p['tags'])}"
        )
    return "\n".join(lines)