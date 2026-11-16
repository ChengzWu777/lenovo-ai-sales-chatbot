"""Per-session answer cache for repeated questions.

A cached answer is only reused when it would still be correct:
  * the customer profile (every constraint) is identical,
  * the catalog has not changed,
  * every number in the question is identical ($1000 vs $2000 must never collide),
  * the negation words are identical ("with AMD" vs "no AMD"),
  * the remaining keywords overlap almost completely (Jaccard >= threshold).

The previous version counted shared keywords (>= 3) and ignored the profile, so
"gaming laptop under $1000" served its answer to "gaming laptop under $2000".
"""
import re
from dataclasses import dataclass, field
from typing import Optional

_STOPWORDS = {
    "i", "a", "an", "the", "is", "are", "do", "you", "have", "what", "which", "can",
    "for", "me", "my", "your", "and", "or", "in", "on", "at", "to", "of", "with",
    "about", "any", "need", "want", "please", "show", "give", "some", "that", "it",
    "would", "like", "looking", "there",
}
_NEGATIONS = {"no", "not", "without", "never", "except", "avoid", "dont", "don", "but"}


@dataclass
class CacheEntry:
    question: str
    answer: str
    category: str
    context: tuple
    keywords: frozenset = field(default_factory=frozenset)
    numbers: frozenset = field(default_factory=frozenset)
    negations: frozenset = field(default_factory=frozenset)
    product_ids: tuple = ()
    hit_count: int = 0


class CacheLayer:
    def __init__(self, similarity_threshold: float = 0.8, max_entries: int = 50):
        self.store: list[CacheEntry] = []
        self.threshold = similarity_threshold
        self.max_entries = max_entries

    @staticmethod
    def _features(text: str) -> tuple[frozenset, frozenset, frozenset]:
        words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower().replace(",", ""))
        numbers = frozenset(w for w in words if any(c.isdigit() for c in w))
        negations = frozenset(w for w in words if w in _NEGATIONS)
        keywords = frozenset(
            w for w in words
            if w not in _STOPWORDS and w not in _NEGATIONS and w not in numbers and len(w) > 1
        )
        return keywords, numbers, negations

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / len(a | b)

    def lookup(self, question: str, context: tuple = ()) -> Optional[CacheEntry]:
        keywords, numbers, negations = self._features(question)
        if not keywords:
            return None
        best, best_score = None, 0.0
        for entry in self.store:
            if entry.context != context or entry.numbers != numbers or entry.negations != negations:
                continue
            score = self._jaccard(keywords, entry.keywords)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.threshold:
            best.hit_count += 1
            return best
        return None

    def store_entry(self, question: str, answer: str, category: str,
                    context: tuple = (), product_ids: tuple = ()):
        # Don't cache short answers (clarifications, errors).
        if len(answer) < 100:
            return
        keywords, numbers, negations = self._features(question)
        if not keywords:
            return
        for entry in self.store:
            if (entry.context == context and entry.numbers == numbers
                    and entry.negations == negations
                    and self._jaccard(keywords, entry.keywords) >= self.threshold):
                return  # an equivalent entry already exists
        self.store.append(CacheEntry(
            question=question, answer=answer, category=category, context=context,
            keywords=keywords, numbers=numbers, negations=negations, product_ids=tuple(product_ids),
        ))
        if len(self.store) > self.max_entries:
            self.store.pop(0)

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self.store),
            "total_hits": sum(e.hit_count for e in self.store),
        }
