"""One implementation of profile constraints, used by every retrieval path.

Hard constraints are never relaxed: budget, rejected processors, rejected lines.
Soft constraints are applied in order and relaxed one by one when nothing would
be left; every relaxation is reported so the answer can say so explicitly
(previously constraints were dropped silently).
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from agents.session_manager import UserProfile

SOFT_BUDGET_TOLERANCE = 1.10   # approximate budgets may be exceeded by 10%


@dataclass
class FilterResult:
    products: list
    relaxed: list = field(default_factory=list)   # human-readable constraints that could not be met
    hard_empty: bool = False                      # no product satisfies the hard constraints


def battery_hours(product: dict) -> Optional[float]:
    m = re.search(r"up to (\d+(?:\.\d+)?)\s*hrs?", product.get("battery", ""))
    return float(m.group(1)) if m else None


def screen_inches(product: dict) -> Optional[float]:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\"", product.get("display", ""))
    return float(m.group(1)) if m else None


def gpu_rank(gpu: str) -> int:
    """Comparable rank for consumer RTX cards; 0 for integrated graphics.

    RTX 4070 -> 40700, RTX 4070 Ti -> 40705. Workstation 'Ada' cards count as
    dedicated (rank 1) but are not ordered against gaming cards.
    """
    g = gpu.lower()
    m = re.search(r"rtx\s*(\d{4})(\s*ti)?", g)
    if not m:
        return 0
    if "ada" in g:
        return 1
    return int(m.group(1)) * 10 + (5 if m.group(2) else 0)


def _gpu_floor(gpu_min: str) -> int:
    if gpu_min == "dedicated":
        return 1
    m = re.match(r"(\d{4})(\s*ti)?", gpu_min)
    return int(m.group(1)) * 10 + (5 if m.group(2) else 0) if m else 1


def _haystack(product: dict) -> str:
    return " ".join([
        product.get("series", ""), product.get("name", ""), product.get("os", ""),
        " ".join(product.get("tags", [])),
    ]).lower()


def budget_ceiling(profile: UserProfile) -> Optional[float]:
    if not profile.budget_max:
        return None
    return profile.budget_max if profile.budget_firm else profile.budget_max * SOFT_BUDGET_TOLERANCE


def passes_hard(product: dict, profile: UserProfile) -> bool:
    ceiling = budget_ceiling(profile)
    if ceiling is not None and product["price"] > ceiling:
        return False
    processor = product.get("processor", "").lower()
    if any(r.lower() in processor for r in profile.processor_reject):
        return False
    hay = _haystack(product)
    if any(r.lower() in hay for r in profile.brand_reject):
        return False
    return True


def _display_ok(product: dict, pref: str) -> bool:
    display = product.get("display", "").lower()
    tags = " ".join(product.get("tags", [])).lower()
    if pref == "oled":
        return "oled" in display
    if pref == "4k":
        return "3840" in display or "4k" in display
    if pref == "2k":
        return any(t in display for t in ("2560", "2880", "3200", "2.8k", "3.2k", "qhd", "wqxga"))
    if pref == "touch":
        return "touch" in display or "touch" in tags
    return True


def _soft_checks(profile: UserProfile) -> list:
    """(label, predicate) pairs in priority order."""
    checks = []
    if profile.series_pref:
        wanted = [s.lower() for s in profile.series_pref]
        checks.append((f"product line {'/'.join(profile.series_pref)}",
                       lambda p: p.get("series", "").lower() in wanted))
    if profile.use_case in ("gaming", "workstation"):
        # A gaming / workstation request should not be padded with office laptops.
        kind = profile.use_case
        checks.append((f"a {kind}-capable laptop (dedicated GPU)",
                       lambda p: p.get("category") == kind or gpu_rank(p.get("gpu", "")) > 0))
    if profile.gpu_min:
        floor = _gpu_floor(profile.gpu_min)
        label = "a dedicated GPU" if profile.gpu_min == "dedicated" else f"RTX {profile.gpu_min.upper()} or better"
        checks.append((label, lambda p: gpu_rank(p.get("gpu", "")) >= floor))
    if profile.weight_max:
        checks.append((f"weight under {profile.weight_max:g} lbs",
                       lambda p: p.get("weight_lbs", 99) <= profile.weight_max))
    if profile.form_factor == "2-in-1":
        checks.append(("2-in-1 form factor",
                       lambda p: any(t.lower() in ("2-in-1", "convertible", "tablet detachable", "detachable")
                                     for t in p.get("tags", []))))
    if profile.display_pref:
        checks.append((f"{profile.display_pref.upper()} display",
                       lambda p: _display_ok(p, profile.display_pref)))
    if profile.battery_min:
        # Unknown battery life does not count as meeting the requirement.
        checks.append((f"{profile.battery_min}+ hours of battery",
                       lambda p: (battery_hours(p) or 0) >= profile.battery_min))
    if profile.screen_size_pref:
        checks.append((f"~{profile.screen_size_pref:g}\" screen",
                       lambda p: abs((screen_inches(p) or 0) - profile.screen_size_pref) <= 0.6))
    if profile.budget_min and profile.budget_max is None:
        checks.append((f"price above ${profile.budget_min}", lambda p: p["price"] >= profile.budget_min))
    return checks


def apply_constraints(products: list, profile: UserProfile) -> FilterResult:
    result = [p for p in products if passes_hard(p, profile)]
    if not result:
        return FilterResult(products=[], hard_empty=True)

    relaxed = []
    for label, predicate in _soft_checks(profile):
        narrowed = [p for p in result if predicate(p)]
        if narrowed:
            result = narrowed
        else:
            relaxed.append(label)
    return FilterResult(products=result, relaxed=relaxed)


def violations(product: dict, profile: UserProfile) -> list[str]:
    """Which profile constraints a product breaks. Used for comparisons and evaluation."""
    out = []
    ceiling = budget_ceiling(profile)
    if ceiling is not None and product["price"] > ceiling:
        out.append(f"over the ${profile.budget_max} budget")
    processor = product.get("processor", "").lower()
    for r in profile.processor_reject:
        if r.lower() in processor:
            out.append(f"uses an excluded processor ({r})")
    hay = _haystack(product)
    for r in profile.brand_reject:
        if r.lower() in hay:
            out.append(f"is an excluded line ({r})")
    for label, predicate in _soft_checks(profile):
        if not predicate(product):
            out.append(f"does not meet: {label}")
    return out
