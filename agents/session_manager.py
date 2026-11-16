import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from agents.cache_layer import CacheLayer


@dataclass
class UserProfile:
    use_case: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    # True for a stated ceiling ("under $1000", "budget is $800").
    # False only for approximate budgets ("around $1200") or "budget is flexible".
    budget_firm: bool = False
    weight_max: Optional[float] = None          # lbs
    brand_reject: list = field(default_factory=list)      # series or terms to exclude
    series_pref: list = field(default_factory=list)       # thinkpad / legion / ...
    processor_pref: Optional[str] = None
    processor_reject: list = field(default_factory=list)
    screen_size_pref: Optional[float] = None    # inches
    display_pref: Optional[str] = None          # oled / 4k / 2k / touch
    gpu_min: Optional[str] = None               # "4070", "4070 ti", "dedicated"
    battery_min: Optional[int] = None           # hours
    form_factor: Optional[str] = None           # "2-in-1"

    def is_empty(self) -> bool:
        return all(v in (None, [], False) for v in asdict(self).values())

    def signature(self) -> tuple:
        """Hashable snapshot of every constraint. Part of the answer-cache key."""
        return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(asdict(self).items()))


@dataclass
class Session:
    session_id: str
    history: list = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    metrics: list = field(default_factory=list)
    cache: CacheLayer = field(default_factory=CacheLayer)
    # Product ids shown in the previous answer, so follow-ups ("the second one",
    # "compare these") stay grounded in the same products.
    last_product_ids: list = field(default_factory=list)
    # True right after the bot asked a clarifying question: the next message is an
    # answer to it, even if it contains no product keywords ("mostly spreadsheets").
    pending_clarify: bool = False


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)


session_manager = SessionManager()
