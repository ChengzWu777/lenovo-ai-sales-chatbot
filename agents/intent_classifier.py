import re
from enum import Enum

from agents.taxonomy import detect_use_case
from agents.text import count_kw


class Intent(str, Enum):
    PRODUCT_INQUIRY = "product"
    SUPPORT_FAQ = "support"
    OFF_TOPIC = "off_topic"
    GREETING = "greeting"
    COMPARISON = "comparison"
    HANDOFF = "handoff"
    ACCESSORY = "accessory"
    THANKS = "thanks"


PRODUCT_KEYWORDS = [
    "laptop", "laptops", "computer", "notebook", "pc", "thinkpad", "thinkbook", "legion", "ideapad",
    "chromebook", "recommend", "suggest", "show me", "looking for", "need", "want", "buy", "purchase",
    "choose", "choosing", "pick", "gaming", "business", "student", "budget", "price", "cost",
    "compare", "difference", "vs", "versus", "best", "cheapest", "lightest", "battery", "display",
    "screen", "gpu", "processor", "cpu", "ram", "memory", "storage", "ssd", "weight", "portable",
    "thin", "oled", "rtx", "intel", "amd", "snapdragon", "workstation", "cad", "rendering",
    "creative", "enterprise", "smb", "travel", "4k", "2-in-1", "convertible", "touch", "specs",
    "model", "models", "lbs",
]

SUPPORT_KEYWORDS = [
    "return", "refund", "warranty claim", "repair", "broken", "defective", "shipping", "delivery",
    "order", "track", "tracking", "cancel", "exchange", "support", "problem", "issue",
    "not working", "crash", "crashing", "crashes", "won't turn on", "policy", "phone number",
    "store location",
]

ACCESSORY_KEYWORDS = [
    "mouse", "mice", "headset", "headphones", "docking station", "docking stations", "dock",
    "monitor", "monitors", "stylus", "pen", "laptop bag", "laptop bags", "backpack",
    "warranty options", "extended warranty", "student discount", "student discounts",
    "bulk pricing", "support plans", "support plan", "keyboard cover", "charger",
]

HANDOFF_PATTERNS = [
    r"\b(?:talk|speak|chat|connect me) (?:to|with) (?:a |an )?(?:\w+ )?(?:rep|representative|human|person|agent|someone)\b",
    r"\b(?:human|live|real) (?:agent|person|rep|representative)\b",
    r"\bsales (?:rep|representative)\b",
]

OFF_TOPIC_KEYWORDS = [
    "weather", "poem", "joke", "recipe", "song", "lyrics", "essay", "story", "translate",
    "stock", "stocks", "crypto", "bitcoin", "election", "president", "politics", "news",
    "movie", "homework answer", "write code", "python script",
]

# Attempts to override the assistant's role never reach the LLM.
INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier|your) (?:instructions|prompts?|rules)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|your) (?:instructions|prompts?|rules)",
    r"(?:reveal|show|print|repeat) (?:me )?(?:your |the )?(?:system|initial|hidden) (?:prompt|instructions)",
    r"\byou are now\b", r"\bpretend (?:to be|you are)\b", r"\bact as (?:a|an|if)\b",
    r"\bjailbreak\b", r"\bdeveloper mode\b", r"\bdan mode\b",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|hiya|good morning|good afternoon|good evening|howdy)( there)?[!.,\s]*$",
    r"^(how are you|how's it going)[?!.,\s]*$",
    r"^what can you (do|help with)[?!.,\s]*$",
]

THANKS_PATTERN = r"^(thanks|thank you|thx|ty|great,? thanks|ok thanks|perfect,? thanks?)( so much| a lot)?[!.\s]*$"


def classify_intent(message: str, has_context: bool = False) -> Intent:
    """Rule-based intent.

    has_context: the session already has a profile or shown products, so a message
    with no keywords ("what about the second one?") is treated as a follow-up.
    Without context such a message is off-topic (previously everything unknown
    defaulted to a product inquiry and reached the LLM).
    """
    msg = message.lower().strip()
    if not msg:
        return Intent.OFF_TOPIC

    if any(re.search(p, msg) for p in INJECTION_PATTERNS):
        return Intent.OFF_TOPIC

    for pattern in GREETING_PATTERNS:
        if re.match(pattern, msg):
            return Intent.GREETING
    if re.match(THANKS_PATTERN, msg):
        return Intent.THANKS

    # A bare number is an answer to "what's your budget?"
    if re.fullmatch(r"\$?\s*[\d,]+(?:\.\d+)?\s*k?", msg):
        return Intent.PRODUCT_INQUIRY

    # Gibberish: no vowels at all in a longer message
    letters = [c for c in msg if c.isalpha()]
    if len(letters) > 3 and not any(c in "aeiouy" for c in letters):
        return Intent.OFF_TOPIC

    if any(re.search(p, msg) for p in HANDOFF_PATTERNS):
        return Intent.HANDOFF

    support_score = count_kw(msg, SUPPORT_KEYWORDS)
    product_score = count_kw(msg, PRODUCT_KEYWORDS)
    # A dollar amount or a recognised use case ("for school", "video editing") is a
    # shopping signal even without the word "laptop".
    if re.search(r"\$\s?\d", msg) or detect_use_case(msg):
        product_score += 1
    accessory_score = count_kw(msg, ACCESSORY_KEYWORDS)
    off_topic_score = count_kw(msg, OFF_TOPIC_KEYWORDS)

    if re.search(r"\bcompare\b|\bvs\.?\b|\bversus\b|difference between|side by side", msg) and (product_score or has_context):
        return Intent.COMPARISON

    if support_score > product_score and support_score >= 1:
        return Intent.SUPPORT_FAQ

    # "what gaming mice do you recommend?" — accessories are not in the laptop catalog.
    if accessory_score and count_kw(msg, ["laptop", "laptops", "thinkpad", "thinkbook", "legion", "ideapad"]) == 0:
        return Intent.ACCESSORY

    if off_topic_score and off_topic_score >= product_score:
        return Intent.OFF_TOPIC

    if product_score >= 1:
        return Intent.PRODUCT_INQUIRY
    if support_score >= 1:
        return Intent.SUPPORT_FAQ

    return Intent.PRODUCT_INQUIRY if has_context else Intent.OFF_TOPIC
