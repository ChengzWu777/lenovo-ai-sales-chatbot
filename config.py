"""Runtime settings, read once from environment variables."""
import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:27b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    # Thinking tokens are generated before the first visible token and can dominate
    # latency on reasoning-capable models. Off by default; set OLLAMA_REASONING=true to compare.
    reasoning: bool = _bool("OLLAMA_REASONING", False)
    # Seconds to wait for the first visible token, and for the whole answer.
    first_token_timeout_s: float = float(os.getenv("LLM_FIRST_TOKEN_TIMEOUT_S", "60"))
    total_timeout_s: float = float(os.getenv("LLM_TOTAL_TIMEOUT_S", "180"))
    history_window: int = int(os.getenv("HISTORY_WINDOW", "6"))
    max_prompt_products: int = int(os.getenv("MAX_PROMPT_PRODUCTS", "8"))
    max_message_chars: int = int(os.getenv("MAX_MESSAGE_CHARS", "2000"))
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = _bool("RELOAD", False)


settings = Settings()
