"""Small text helpers shared by the deterministic modules."""
import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=4096)
def _kw_pattern(keyword: str) -> re.Pattern:
    # Word-boundary match that also works for keywords containing spaces or
    # punctuation ("2-in-1", "core ultra"). Prevents "ram" matching "program".
    return re.compile(rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])")


def has_kw(msg: str, keyword: str) -> bool:
    """True if keyword occurs in msg as a whole word/phrase. msg must be lowercase."""
    return _kw_pattern(keyword).search(msg) is not None


def any_kw(msg: str, keywords) -> bool:
    return any(has_kw(msg, kw) for kw in keywords)


def count_kw(msg: str, keywords) -> int:
    return sum(1 for kw in keywords if has_kw(msg, kw))


# Hard negation: the user rules something out.
_NEG_HARD = (
    r"(?:no|not|non|without|avoid|avoiding|never|except|excluding|hate|dislike|"
    r"don'?t\s+(?:want|like|need)|do\s+not\s+(?:want|like|need)|"
    r"anything\s+but|other\s+than|nothing\s+(?:from|with))"
)
# Soft negation: the user deprioritises something ("Intel over AMD", "forget gaming").
_NEG_SOFT = r"(?:over|instead\s+of|rather\s+than|forget|forget\s+about|skip|than)"
_FILLER = r"(?:\s+(?:an?|any|the|want|like|need|more|that|those))*"


def is_rejected(msg: str, keyword: str) -> bool:
    """'no AMD', 'do NOT want AMD', 'anything but ideapad' -> True."""
    return re.search(
        rf"(?<![a-z0-9]){_NEG_HARD}{_FILLER}\s+{re.escape(keyword)}(?![a-z0-9])", msg
    ) is not None


def is_deprioritised(msg: str, keyword: str) -> bool:
    """'Intel over AMD', 'forget gaming' -> True for amd / gaming."""
    return re.search(
        rf"(?<![a-z0-9]){_NEG_SOFT}{_FILLER}\s+{re.escape(keyword)}(?![a-z0-9])", msg
    ) is not None
