from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from agents.session_manager import Session
from agents.intent_classifier import classify_intent, Intent
from agents.slot_filler import extract_slots, get_missing_slots, format_profile_summary
from agents.query_router import route_query, format_products_for_prompt
from agents.product_matcher import match_products, format_factual_response

# ── State ──
class ChatState(TypedDict):
    session: Session
    user_msg: str
    # Outputs
    action: str          # "direct" | "clarify" | "factual" | "llm" | "cache"
    response: Optional[str]        # for direct/clarify/factual/cache
    category: Optional[str]
    matched_products: Optional[list]
    system_prompt: Optional[str]
    skip: bool

CLARIFY_USE_CASE = "I'd love to help you find the perfect Lenovo laptop! Could you tell me what you'll mainly use it for — gaming, business, creative work, or general everyday use?"
SUPPORT_RESPONSE = "For order tracking, returns, warranty claims, or technical support, please visit **support.lenovo.com** or call **1-855-253-6686**. Is there anything else I can help you with today?"
OFF_TOPIC_RESPONSE = "I didn't quite catch that! I'm here to help you find the perfect Lenovo laptop. You can ask me about gaming, business, student, or creative laptops — or just tell me your budget and I'll find the best match."
SYSTEM_PROMPT_TEMPLATE = """You are an expert Lenovo sales assistant.

CUSTOMER PROFILE:
{profile_summary}

RELEVANT PRODUCTS (pre-filtered for this query):
{product_catalog}

INSTRUCTIONS:
- Recommend ONLY products from the list above
- Always cite exact prices and specs from the list
- Be concise, friendly, and professional
- If the customer has a firm budget, never suggest anything above it
- Always respond in English"""

# ── Nodes ──

def node_intent(state: ChatState) -> ChatState:
    intent = classify_intent(state["user_msg"])
    state["skip"] = False

    if intent == Intent.GREETING:
        state["action"] = "direct"
        state["response"] = "Hi! I'm your Lenovo AI Sales Assistant. What are you looking for today?"
        state["skip"] = True
    elif intent == Intent.SUPPORT_FAQ:
        state["action"] = "direct"
        state["response"] = SUPPORT_RESPONSE
        state["skip"] = True
    elif intent == Intent.OFF_TOPIC:
        state["action"] = "direct"
        state["response"] = OFF_TOPIC_RESPONSE
        state["skip"] = True
    return state


def node_cache(state: ChatState) -> ChatState:
    if state["skip"]:
        return state
    hit = state["session"].cache.lookup(state["user_msg"])
    if hit:
        state["action"] = "cache"
        state["response"] = hit.answer
        state["category"] = hit.category
        state["skip"] = True
    return state


def node_slots(state: ChatState) -> ChatState:
    if state["skip"]:
        return state
    state["session"].profile = extract_slots(state["user_msg"], state["session"].profile)
    return state


def node_clarify(state: ChatState) -> ChatState:
    if state["skip"]:
        return state
    missing = get_missing_slots(state["session"].profile)
    cross_category = any(kw in state["user_msg"].lower() for kw in [
        "cheapest", "most affordable", "lowest price", "least expensive",
        "show me all", "list all", "2-in-1", "convertible",
        "show me thinkpad", "show me legion", "thinkpad options", "legion options",
        "i want a legion", "i want a thinkpad", "legion laptop", "thinkpad laptop"
    ])
    budget_only = state["session"].profile.budget_max is not None and not state["session"].profile.use_case

    if "use_case" in missing and state["session"].turn_count == 1 and not cross_category and not budget_only:
        state["action"] = "clarify"
        state["response"] = CLARIFY_USE_CASE
        state["skip"] = True
    return state


def node_router(state: ChatState) -> ChatState:
    if state["skip"]:
        return state

    from agents.rag_retriever import get_retriever
    retriever = get_retriever()

    # RAG retrieval
    products = retriever.retrieve(state["user_msg"], state["session"].profile, n_results=12)

    # Fallback: if RAG returns too few, use keyword router
    if len(products) < 3:
        from agents.query_router import route_query
        category, products = route_query(state["user_msg"], state["session"].profile)
        state["category"] = category
    else:
        # Infer category from top result
        top = products[0] if products else None
        # Infer category from query keywords, not just product series
        msg = state["user_msg"].lower()
        if any(kw in msg for kw in ["gaming", "game", "fps", "legion", "rtx"]):
            state["category"] = "gaming"
        elif any(kw in msg for kw in ["business", "enterprise", "thinkpad", "office", "work"]):
            state["category"] = "business"
        elif any(kw in msg for kw in ["creative", "video editing", "photo", "design", "oled", "color", "rendering", "3d"]):
            state["category"] = "creative"
        elif any(kw in msg for kw in ["student", "school", "college", "homework"]):
            state["category"] = "student"
        elif any(kw in msg for kw in ["thinkbook", "smb", "startup", "convertible", "2-in-1"]):
            state["category"] = "smb"
        elif top:
            series_to_cat = {
                "Legion": "gaming", "ThinkPad": "business",
                "ThinkBook": "smb", "IdeaPad": "all"
            }
            state["category"] = series_to_cat.get(top.get("series", ""), "all")
        else:
            state["category"] = "all"

    # Safety: reset bad budget parse
    if len(products) == 0:
        profile = state["session"].profile
        if profile.budget_max and profile.budget_max < 200:
            profile.budget_max = None
            profile.budget_min = None
            products = retriever.retrieve(state["user_msg"], profile, n_results=12)

    from agents.product_matcher import match_products, format_factual_response
    matched, needs_llm = match_products(
        state["user_msg"], state["session"].profile, products
    )
    state["matched_products"] = matched

    # Factual shortcut
    if not needs_llm and matched:
        factual = format_factual_response(matched, state["user_msg"])
        if factual:
            state["action"] = "factual"
            state["response"] = factual
            state["skip"] = True
            return state

    # Build system prompt for LLM
    from agents.slot_filler import format_profile_summary
    from agents.query_router import format_products_for_prompt
    profile_summary = format_profile_summary(state["session"].profile)
    product_catalog = format_products_for_prompt(matched, max_products=12)
    state["system_prompt"] = SYSTEM_PROMPT_TEMPLATE.format(
        profile_summary=profile_summary,
        product_catalog=product_catalog
    )
    state["action"] = "llm"
    return state


# ── Edges ──
def after_intent(state: ChatState) -> str:
    return "end" if state["skip"] else "cache"

def after_cache(state: ChatState) -> str:
    return "end" if state["skip"] else "slots"

def after_clarify(state: ChatState) -> str:
    return "end" if state["skip"] else "router"

def after_router(state: ChatState) -> str:
    return "end"


# ── Build ──
def build_sales_graph():
    g = StateGraph(ChatState)
    g.add_node("intent", node_intent)
    g.add_node("cache", node_cache)
    g.add_node("slots", node_slots)
    g.add_node("clarify", node_clarify)
    g.add_node("router", node_router)

    g.set_entry_point("intent")
    g.add_conditional_edges("intent", after_intent, {"end": END, "cache": "cache"})
    g.add_conditional_edges("cache", after_cache, {"end": END, "slots": "slots"})
    g.add_edge("slots", "clarify")
    g.add_conditional_edges("clarify", after_clarify, {"end": END, "router": "router"})
    g.add_conditional_edges("router", after_router, {"end": END})

    return g.compile()