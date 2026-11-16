"""Pure check functions used by the evaluator. No LLM, no I/O — unit-tested in tests/."""
import re

from agents.catalog import find_exact_mentions
from agents.filters import SOFT_BUDGET_TOLERANCE

_PRICE_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?")


def extract_prices(text: str) -> list[int]:
    """Dollar amounts in text, including '$1,299' (the old regex only saw '$1299')."""
    return [int(m.group(1).replace(",", "")) for m in _PRICE_RE.finditer(text)]


def check_keywords(text: str, any_of=(), all_of=(), none_of=()) -> dict:
    low = text.lower()
    missing_all = [k for k in all_of if k.lower() not in low]
    found_forbidden = [k for k in none_of if k.lower() in low]
    any_ok = (not any_of) or any(k.lower() in low for k in any_of)
    return {
        "ok": any_ok and not missing_all and not found_forbidden,
        "missing_any": list(any_of) if not any_ok else [],
        "missing_all": missing_all,
        "found_forbidden": found_forbidden,
    }


def check_constraints(products: list, max_price=None, reject_processor=(), reject_terms=()) -> dict:
    """Do these catalog products respect the case's hard constraints?

    Uses catalog facts, not text matching, so answer formatting cannot hide a violation.
    """
    problems = []
    for p in products:
        if max_price is not None and p["price"] > max_price:
            problems.append(f"{p['name']} costs ${p['price']} > ${max_price:g}")
        for r in reject_processor:
            if r.lower() in p["processor"].lower():
                problems.append(f"{p['name']} uses {r}")
        hay = f"{p['series']} {p['name']} {p.get('os', '')} {' '.join(p.get('tags', []))}".lower()
        for r in reject_terms:
            if r.lower() in hay:
                problems.append(f"{p['name']} is {r}")
    return {"ok": not problems, "problems": problems}


def check_grounding(response: str, provided: list) -> dict:
    """Is the answer grounded in the products the LLM was given?

    * every catalog product named in full must be one that was provided
    * a dollar amount right after a product name must equal its catalog price
    """
    provided_ids = {p["id"] for p in provided}
    mentioned = find_exact_mentions(response)
    ungrounded = [p["name"] for p in mentioned if p["id"] not in provided_ids]

    wrong_prices = []
    low = response.lower().replace("*", "")
    for p in mentioned:
        name = p["name"].lower()
        start = 0
        while True:
            idx = low.find(name, start)
            if idx < 0:
                break
            end = idx + len(name)
            start = end
            window = low[end:end + 60]
            # Stop at the next line: a table row or bullet belongs to one product.
            window = window.split("\n")[0]
            prices = extract_prices(window)
            if prices and prices[0] != p["price"]:
                # "X ($1,899) is $400 more than ..." -> only the first amount is the price claim
                wrong_prices.append(f"{p['name']}: said ${prices[0]}, catalog ${p['price']}")
    return {
        "ok": not ungrounded and not wrong_prices,
        "mentioned": [p["name"] for p in mentioned],
        "ungrounded": ungrounded,
        "wrong_prices": wrong_prices,
    }


def budget_ceiling(budget_max, firm: bool):
    if budget_max is None:
        return None
    return budget_max if firm else budget_max * SOFT_BUDGET_TOLERANCE


def check_profile(profile, expected: dict) -> dict:
    """Compare selected profile fields with their expected values."""
    wrong = {}
    for key, value in expected.items():
        actual = getattr(profile, key)
        if actual != value:
            wrong[key] = {"expected": value, "actual": actual}
    return {"ok": not wrong, "wrong": wrong}
