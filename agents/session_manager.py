from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class UserProfile:
    use_case: Optional[str] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    budget_firm: bool = False
    weight_max: Optional[float] = None
    brand_reject: list = field(default_factory=list)
    processor_pref: Optional[str] = None
    processor_reject: list = field(default_factory=list)
    screen_size_pref: Optional[str] = None
    display_pref: Optional[str] = None
    gpu_min: Optional[str] = None
    battery_min: Optional[int] = None
    form_factor: Optional[str] = None
    routed_category: Optional[str] = None

@dataclass
class Session:
    session_id: str
    history: list = field(default_factory=list)
    profile: UserProfile = field(default_factory=UserProfile)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    metrics: list = field(default_factory=list)
    cache: object = None

    def __post_init__(self):
        from agents.cache_layer import CacheLayer
        self.cache = CacheLayer()

class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)
        return self.sessions[session_id]

    def clear(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()