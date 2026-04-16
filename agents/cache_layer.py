import re
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CacheEntry:
    question: str
    answer: str
    category: str
    keywords: set = field(default_factory=set)
    hit_count: int = 0

class CacheLayer:
    def __init__(self, similarity_threshold: int = 3):
        self.store: list[CacheEntry] = []
        self.threshold = similarity_threshold

    def _extract_keywords(self, text: str) -> set:
        text = text.lower()
        stopwords = {"i", "a", "an", "the", "is", "are", "do", "you", "have",
                     "what", "which", "can", "for", "me", "my", "your", "and",
                     "or", "in", "on", "at", "to", "of", "with", "about", "any"}
        words = re.findall(r'\b\w+\b', text)
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _similarity(self, kw1: set, kw2: set) -> int:
        return len(kw1 & kw2)

    def lookup(self, question: str) -> Optional[CacheEntry]:
        q_keywords = self._extract_keywords(question)
        best_match = None
        best_score = 0

        for entry in self.store:
            score = self._similarity(q_keywords, entry.keywords)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_score >= self.threshold:
            best_match.hit_count += 1
            return best_match
        return None

    def store_entry(self, question: str, answer: str, category: str):
        keywords = self._extract_keywords(question)
        # Don't cache very short answers or clarification responses
        if len(answer) < 100:
            return
        # Don't cache if similar entry already exists
        existing = self.lookup(question)
        if existing and self._similarity(keywords, existing.keywords) >= self.threshold + 1:
            return
        entry = CacheEntry(
            question=question,
            answer=answer,
            category=category,
            keywords=keywords
        )
        self.store.append(entry)

    def get_stats(self) -> dict:
        return {
            "total_entries": len(self.store),
            "total_hits": sum(e.hit_count for e in self.store),
        }

# Per-session cache — instantiated in session_manager