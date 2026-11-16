"""Ranking, superlative handling, and deterministic (no-LLM) answers."""
import re
from typing import Optional

from agents.filters import battery_hours, gpu_rank
from agents.session_manager import UserProfile
from agents.text import has_kw

RRF_K = 60

FACTUAL_PATTERNS = [
    r"how much (?:is|does|for)", r"what(?:'s| is)? (?:the )?(?:price|cost)", r"\bprice of\b",
    r"how heavy", r"how much does .* weigh", r"what(?:'s| is)? (?:the )?weight", r"\bweigh\b",
    r"how long .*battery", r"what(?:'s| is)? (?:the )?battery",
    r"what(?:'s| is)? (?:the )?(?:display|screen)", r"what ports", r"which ports",
    r"what(?:'s| is)? (?:the )?(?:processor|cpu|gpu|graphics|ram|memory|storage)",
    r"\b(?:full )?specs?\b", r"\bspecifications\b",
]
LIST_PATTERNS = [r"\bshow me all\b", r"\blist (?:all|every)\b", r"\bshow all\b", r"\ball (?:the |your )?\w+ (?:laptops|models)\b"]
COMPARISON_PATTERNS = [r"\bvs\.?\b", r"\bversus\b", r"\bcompare\b", r"\bcomparison\b", r"difference between",
                       r"which (?:is|one is) better", r"\bside by side\b"]
RECOMMEND_PATTERNS = [r"recommend", r"suggest", r"\bbest\b", r"help me (?:choose|decide|pick)", r"should i"]

# (pattern, label, sort key, reverse)
SUPERLATIVES = [
    (r"\bcheapest\b|most affordable|lowest[- ]price|least expensive|most budget[- ]friendly",
     "price, lowest first", lambda p: p["price"], False),
    (r"\blightest\b|most portable|lowest weight",
     "weight, lightest first", lambda p: p.get("weight_lbs", 99), False),
    (r"(?:best|longest|most) battery|battery lasts? (?:the )?longest",
     "battery life, longest first", lambda p: battery_hours(p) or 0, True),
    (r"most powerful|best performance|\bfastest\b|top of the line|highest[- ]end|most performance",
     "performance (GPU tier, then price), highest first", lambda p: (gpu_rank(p.get("gpu", "")), p["price"]), True),
    (r"most expensive|\bpriciest\b",
     "price, highest first", lambda p: p["price"], True),
]


def _any(patterns, msg: str) -> bool:
    return any(re.search(p, msg) for p in patterns)


def is_factual(msg: str) -> bool:
    return _any(FACTUAL_PATTERNS, msg) and not _any(RECOMMEND_PATTERNS, msg)


def is_list_request(msg: str) -> bool:
    return _any(LIST_PATTERNS, msg)


def is_comparison(msg: str) -> bool:
    return _any(COMPARISON_PATTERNS, msg)


def detect_superlative(msg: str) -> Optional[tuple]:
    """Returns (label, key, reverse) for 'cheapest', 'lightest', 'best battery' ..."""
    for pattern, label, key, reverse in SUPERLATIVES:
        if re.search(pattern, msg):
            return label, key, reverse
    return None


def heuristic_score(product: dict, profile: UserProfile, msg: str) -> float:
    score = 0.0
    if profile.budget_max:
        ratio = product["price"] / profile.budget_max
        if 0.7 <= ratio <= 1.0:
            score += 3
        elif 0.5 <= ratio < 0.7:
            score += 1
        if profile.budget_min and product["price"] < profile.budget_min:
            score -= 1
    for tag in product.get("tags", []):
        if has_kw(msg, tag.lower()):
            score += 2
    if profile.use_case:
        uc = profile.use_case.lower()
        if any(uc in u.lower() for u in product.get("use_cases", [])) or uc == product.get("category"):
            score += 2
    if profile.processor_pref and profile.processor_pref.lower() in product.get("processor", "").lower():
        score += 2
    desc = product.get("description", "").lower()
    score += sum(1 for word in set(msg.split()) if len(word) > 4 and has_kw(desc, word))
    return score


def rank_products(products: list, profile: UserProfile, message: str,
                  semantic_ids: Optional[list] = None) -> list:
    """Order candidates that already satisfy the profile constraints.

    Reciprocal-rank fusion of the semantic order (vector search) and a lexical /
    profile heuristic. Previously the heuristic re-sort discarded the vector order.
    """
    msg = message.lower()
    by_heuristic = sorted(products, key=lambda p: (-heuristic_score(p, profile, msg), p["price"]))
    if not semantic_ids:
        return by_heuristic
    sem_rank = {pid: i for i, pid in enumerate(semantic_ids)}
    heu_rank = {p["id"]: i for i, p in enumerate(by_heuristic)}
    missing = len(products)

    def fused(p):
        return (1 / (RRF_K + sem_rank.get(p["id"], missing)) +
                1 / (RRF_K + heu_rank[p["id"]]))

    return sorted(products, key=lambda p: (-fused(p), p["price"]))


def format_product_line(p: dict) -> str:
    return (
        f"- {p['name']} | ${p['price']} | {p['processor']} | {p['gpu']} | {p['ram']} | "
        f"{p.get('storage', 'n/a')} | {p['display']} | {p['weight_lbs']} lbs | "
        f"Battery: {p['battery']} | OS: {p.get('os', 'n/a')} | Tags: {', '.join(p['tags'])}"
    )


def format_products_for_prompt(products: list, max_products: int = 8) -> str:
    return "\n".join(format_product_line(p) for p in products[:max_products])


def format_spec_card(p: dict) -> str:
    rows = [
        ("Price", f"${p['price']}"), ("Processor", p["processor"]), ("Graphics", p["gpu"]),
        ("Memory", p["ram"]), ("Storage", p.get("storage")), ("Display", p["display"]),
        ("Weight", f"{p['weight_lbs']} lbs ({p.get('weight_kg', '?')} kg)"),
        ("Battery", p["battery"]), ("Ports", p.get("ports")), ("OS", p.get("os")),
    ]
    lines = [f"**{p['name']}**", ""]
    lines += [f"- **{label}:** {value}" for label, value in rows if value]
    return "\n".join(lines)


def format_factual_response(product: dict, message: str) -> Optional[str]:
    """One-line factual answer about a single named product, straight from the catalog."""
    msg = message.lower()
    p = product
    if re.search(r"\bspecs?\b|specifications", msg):
        return format_spec_card(p)
    if re.search(r"\bprice\b|\bcost\b|how much (?:is|for)|how much does .* cost", msg):
        return f"The **{p['name']}** is priced at **${p['price']}**."
    if re.search(r"weigh|heavy", msg):
        return f"The **{p['name']}** weighs **{p['weight_lbs']} lbs** ({p['weight_kg']} kg)."
    if "battery" in msg:
        return f"The **{p['name']}** battery: {p['battery']}."
    if re.search(r"display|screen", msg):
        return f"The **{p['name']}** display: {p['display']}."
    if "port" in msg:
        return f"The **{p['name']}** ports: {p['ports']}."
    if re.search(r"processor|cpu", msg):
        return f"The **{p['name']}** uses the **{p['processor']}**."
    if re.search(r"gpu|graphics", msg):
        return f"The **{p['name']}** graphics: **{p['gpu']}**."
    if re.search(r"\bram\b|memory", msg):
        return f"The **{p['name']}** comes with **{p['ram']}**."
    if "storage" in msg:
        return f"The **{p['name']}** storage: **{p['storage']}**."
    return None


def format_list_response(products: list, note: str = "") -> str:
    lines = [f"Here are the {len(products)} matching laptops:" if len(products) != 1 else "Here is the one matching laptop:", ""]
    for p in sorted(products, key=lambda p: p["price"]):
        lines.append(f"- **{p['name']}** — ${p['price']} | {p['processor']} | {p['gpu']} | {p['weight_lbs']} lbs")
    if note:
        lines += ["", note]
    return "\n".join(lines)
