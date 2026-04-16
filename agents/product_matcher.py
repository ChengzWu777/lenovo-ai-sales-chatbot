import json
import os
import re
from agents.session_manager import UserProfile

PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")

def load_products() -> list:
    with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["products"]

def match_products(message: str, profile: UserProfile, routed_products: list) -> tuple[list, bool]:
    """
    Returns (matched_products, needs_llm)
    needs_llm = False means we can answer directly from DB (factual query)
    needs_llm = True means we need LLM for explanation/comparison
    """
    msg = message.lower()

    # Detect factual queries that don't need LLM
    factual_patterns = [
        r"how much (is|does|cost)",
        r"what (is the )?price",
        r"how heavy",
        r"how much (does it weigh|is it)",
        r"what.*weight",
        r"how long.*battery",
        r"what.*battery",
        r"what.*display",
        r"what.*screen",
        r"does it have",
        r"what ports",
        r"show me all",
        r"list (all|every)",
    ]

    is_factual = any(re.search(p, msg) for p in factual_patterns)

    # Detect comparison queries (always need LLM)
    comparison_patterns = [
        r"\bvs\b", r"versus", r"compare", r"difference between",
        r"which (is|one) better", r"should i (get|buy|choose)"
    ]
    is_comparison = any(re.search(p, msg) for p in comparison_patterns)

    # Detect recommendation queries (need LLM)
    recommendation_patterns = [
        r"recommend", r"suggest", r"what.*best",
        r"which.*best", r"help me (choose|decide|pick)",
        r"what (would you|do you) recommend"
    ]
    is_recommendation = any(re.search(p, msg) for p in recommendation_patterns)

    needs_llm = is_comparison or is_recommendation or not is_factual

    # Check if asking about specific product by name
    specific_product = find_specific_product(msg, routed_products)
    if specific_product and is_factual:
        return [specific_product], False

    # Sort by relevance
    scored = score_products(routed_products, profile, msg)
    top_products = scored[:10]

    return top_products, needs_llm


def find_specific_product(message: str, products: list):
    msg = message.lower()
    for product in products:
        name_lower = product["name"].lower()
        product_id = product["id"].lower()
        if name_lower in msg or product_id in msg:
            return product
        # Check partial name match
        words = name_lower.split()
        if len(words) >= 3 and all(w in msg for w in words[-3:]):
            return product
    return None


def score_products(products: list, profile: UserProfile, message: str) -> list:
    msg = message.lower()
    scored = []

    for p in products:
        score = 0

        # Price score — closer to budget max is better
        if profile.budget_max:
            price_ratio = p["price"] / profile.budget_max
            if 0.7 <= price_ratio <= 1.0:
                score += 3
            elif 0.5 <= price_ratio < 0.7:
                score += 1

        # Tag match
        for tag in p.get("tags", []):
            if tag.lower() in msg:
                score += 2

        # Use case match
        for use_case in p.get("use_cases", []):
            if profile.use_case and profile.use_case.lower() in use_case.lower():
                score += 2

        # Description keyword match
        desc = p.get("description", "").lower()
        for word in msg.split():
            if len(word) > 4 and word in desc:
                score += 1

        scored.append((score, p))

    scored.sort(key=lambda x: (-x[0], x[1]["price"]))
    return [p for _, p in scored]


def format_factual_response(products: list, message: str) -> str:
    msg = message.lower()

    if len(products) == 1:
        p = products[0]
        if any(kw in msg for kw in ["price", "cost", "much"]):
            return f"The **{p['name']}** is priced at **${p['price']}**."
        if any(kw in msg for kw in ["weight", "heavy", "weigh"]):
            return f"The **{p['name']}** weighs **{p['weight_lbs']} lbs** ({p['weight_kg']} kg)."
        if any(kw in msg for kw in ["battery"]):
            return f"The **{p['name']}** has a {p['battery']}."
        if any(kw in msg for kw in ["display", "screen"]):
            return f"The **{p['name']}** has a {p['display']}."
        if any(kw in msg for kw in ["ports"]):
            return f"The **{p['name']}** ports: {p['ports']}."

    if "show me all" in msg or "list" in msg:
        lines = [f"Here are the matching laptops:\n"]
        for p in products:
            lines.append(f"- **{p['name']}** — ${p['price']} | {p['processor']} | {p['gpu']}")
        return "\n".join(lines)

    return None