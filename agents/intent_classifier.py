from enum import Enum
import re

class Intent(str, Enum):
    PRODUCT_INQUIRY = "product"
    SUPPORT_FAQ = "support"
    OFF_TOPIC = "off_topic"
    GREETING = "greeting"
    COMPARISON = "comparison"

PRODUCT_KEYWORDS = [
    "laptop", "computer", "thinkpad", "thinkbook", "legion", "ideapad",
    "recommend", "suggest", "show me", "looking for", "need a", "want a",
    "gaming", "business", "student", "budget", "price", "cost", "buy",
    "compare", "difference", "vs", "versus", "best", "cheapest", "lightest",
    "battery", "display", "screen", "gpu", "processor", "cpu", "ram", "memory",
    "storage", "ssd", "weight", "portable", "thin", "oled", "rtx", "intel",
    "amd", "snapdragon", "workstation", "cad", "rendering", "creative",
    "enterprise", "smb", "travel", "4k", "2-in-1", "convertible", "touch"
]

SUPPORT_KEYWORDS = [
    "return", "refund", "warranty", "repair", "broken", "defective",
    "shipping", "delivery", "order", "track", "cancel", "exchange",
    "support", "help", "problem", "issue", "not working", "policy",
    "contact", "phone number", "email", "store", "location"
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening|howdy)[!.,\s]*$",
    r"^(how are you|how's it going)[?!.,\s]*$",
    r"^what can you (do|help with)[?!.,\s]*$",
]

def classify_intent(message: str) -> Intent:
    msg = message.lower().strip()

    # Reject very short or meaningless messages
    if len(msg) <= 2:
        return Intent.OFF_TOPIC

    # Reject gibberish — no vowels and length > 3 (but allow pure numbers as budget input)
    vowels = set('aeiou')
    if len(msg) > 3 and not msg.isdigit() and not any(c in vowels for c in msg):
        return Intent.OFF_TOPIC

    for pattern in GREETING_PATTERNS:
        if re.match(pattern, msg):
            return Intent.GREETING

    support_score = sum(1 for kw in SUPPORT_KEYWORDS if kw in msg)
    product_score = sum(1 for kw in PRODUCT_KEYWORDS if kw in msg)

    if ("compare" in msg or " vs " in msg or "versus" in msg) and product_score > 0:
        return Intent.COMPARISON

    if support_score > product_score and support_score >= 2:
        return Intent.SUPPORT_FAQ

    if product_score >= 1:
        return Intent.PRODUCT_INQUIRY

    if support_score >= 1:
        return Intent.SUPPORT_FAQ

    return Intent.PRODUCT_INQUIRY