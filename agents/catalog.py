"""Product catalog access: cached load, version hash, and exact model-name lookup."""
import hashlib
import json
import os
import re
import threading
from typing import Optional

from agents.text import tokenize

PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")

_lock = threading.Lock()
_cache: dict = {"mtime": None, "products": None, "version": None, "index": None}


def _load() -> dict:
    """Reload only when the file changes on disk (was: re-read on every request)."""
    mtime = os.path.getmtime(PRODUCTS_PATH)
    with _lock:
        if _cache["mtime"] != mtime:
            with open(PRODUCTS_PATH, "rb") as f:
                raw = f.read()
            _cache["products"] = json.loads(raw.decode("utf-8"))["products"]
            _cache["version"] = hashlib.sha1(raw).hexdigest()[:12]
            _cache["index"] = [_name_parts(p) for p in _cache["products"]]
            _cache["mtime"] = mtime
        return _cache


def load_products() -> list:
    return _load()["products"]


def catalog_version() -> str:
    """Short content hash. Used to invalidate the vector index and the answer cache."""
    return _load()["version"]


def get_by_id(product_id: str) -> Optional[dict]:
    for p in load_products():
        if p["id"] == product_id:
            return p
    return None


# ── Model-name matching ──────────────────────────────────────────────────────
# A product name is split into:
#   core       tokens before the first "Gen", screen size or parenthesis
#              ("ThinkPad X1 Carbon Gen 13 Aura" -> thinkpad x1 carbon)
#   gen        generation number, if any
#   qualifiers everything else (aura, amd, 16, oled, refresh ...)
# A message mentions a product when the core appears as a contiguous token run.
# Among products sharing a core, pick by generation / qualifiers named in the
# message, defaulting to the newest base model.

def _name_parts(product: dict) -> dict:
    name = product["name"].lower()
    gen_match = re.search(r"\bgen\s*(\d+)", name)
    gen = int(gen_match.group(1)) if gen_match else None
    cut = len(name)
    for pattern in (r"\bgen\b", r"\d+(?:\.\d+)?\"", r"\("):
        m = re.search(pattern, name)
        if m:
            cut = min(cut, m.start())
    core = tokenize(name[:cut])
    rest = re.sub(r"\bgen\s*\d+\+?", " ", name[cut:])
    qualifiers = [t for t in tokenize(rest) if not re.fullmatch(r"20[12]\d", t)]
    return {
        "product": product, "core": core, "gen": gen, "qualifiers": qualifiers,
        "full": tokenize(product["name"]),
    }


def _find_runs(tokens: list[str], run: list[str]) -> list[tuple[int, int]]:
    n = len(run)
    if n == 0:
        return []
    return [(i, i + n) for i in range(len(tokens) - n + 1) if tokens[i:i + n] == run]


def _find_run(tokens: list[str], run: list[str]) -> Optional[tuple[int, int]]:
    runs = _find_runs(tokens, run)
    return runs[0] if runs else None


def _distinctive(token: str) -> bool:
    # "t14s", "x13s", "z13" are safe single-token aliases; "5i", "go", "plus" are not.
    return len(token) >= 3 and any(c.isdigit() for c in token) and any(c.isalpha() for c in token)


_WINDOW = 5  # tokens after a model mention that may qualify it ("... Gen 12", "... AMD")


def find_products_by_name(message: str) -> list[dict]:
    """Catalog products explicitly named in the message, in order of mention."""
    tokens = tokenize(message)
    if not tokens:
        return []

    # 1. Every occurrence of every core.
    matches = []  # (span, parts)
    for parts in _load()["index"]:
        core = parts["core"]
        spans = _find_runs(tokens, core)
        if len(core) >= 2:
            tail = core[1:]  # allow dropping the series: "X1 Carbon", "T14s"
            if len(tail) >= 2 or _distinctive(tail[0]):
                spans = spans + [s for s in _find_runs(tokens, tail)
                                 if not any(o[0] <= s[0] and s[1] <= o[1] for o in spans)]
        matches.extend((span, parts) for span in spans)
    if not matches:
        return []

    # 2. Drop matches whose span sits inside a longer one ("legion 5" inside "legion pro 5").
    def contained(a, b):
        return b[0] <= a[0] and a[1] <= b[1] and (b[1] - b[0]) > (a[1] - a[0])

    all_spans = {m[0] for m in matches}
    matches = [m for m in matches if not any(contained(m[0], s) for s in all_spans)]

    # 3. Group by mention (overlapping spans), choose one product per mention using
    #    the tokens that follow it.
    groups: dict[tuple, list] = {}
    for span, parts in matches:
        key = next((k for k in groups if not (span[1] <= k[0] or k[1] <= span[0])), span)
        groups.setdefault(key, []).append(parts)

    ordered = sorted(groups)
    chosen: list[dict] = []
    for idx, span in enumerate(ordered):
        end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(tokens)
        window = tokens[span[1]:min(end, span[1] + _WINDOW)]
        gens = {int(window[i + 1]) for i in range(len(window) - 1)
                if window[i] == "gen" and window[i + 1].isdigit()}
        window_set = set(window)

        def rank(parts):
            quals = parts["qualifiers"]
            overlap = sum(1 for q in quals if q in window_set)
            return (
                parts["gen"] in gens,
                overlap,
                -(len(quals) - overlap),   # fewer unmentioned qualifiers = base model
                parts["gen"] or 0,          # newest generation
            )

        product = max(groups[span], key=rank)["product"]
        if product not in chosen:
            chosen.append(product)
    return chosen


def find_exact_mentions(text: str) -> list[dict]:
    """Products whose full name appears verbatim (token-wise) in text. Used by evaluation."""
    tokens = tokenize(text)
    found = []
    for parts in _load()["index"]:
        if _find_run(tokens, parts["full"]):
            found.append(parts["product"])
    # Prefer the longest names: "X1 Carbon Gen 13 Aura" should not also count as "Gen 13".
    result = []
    for p in found:
        full = tokenize(p["name"])
        longer = [q for q in found if q is not p and _find_run(tokenize(q["name"]), full)]
        if longer and all(_find_run(tokens, tokenize(q["name"])) for q in longer):
            # Only drop the shorter name if every occurrence is explained by a longer one.
            count_short = sum(1 for i in range(len(tokens)) if tokens[i:i + len(full)] == full)
            count_long = sum(
                sum(1 for i in range(len(tokens)) if tokens[i:i + len(tokenize(q["name"]))] == tokenize(q["name"]))
                for q in longer
            )
            if count_short <= count_long:
                continue
        result.append(p)
    return result
