from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import uvicorn
import json
import time
import uuid

from agents.session_manager import session_manager
from agents.sales_graph import build_sales_graph

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

llm = ChatOllama(
    model="qwen3.5:27b",
    base_url="http://localhost:11434",
    temperature=0.7,
)

graph = build_sales_graph()

def build_cta(category: str) -> str:
    cta_data = {
        "actions": [
            {"label": "View Full Specs", "action": "specs"},
            {"label": "Compare Models", "action": "compare"},
            {"label": "Talk to a Rep", "action": "rep"},
        ],
        "crosssell": []
    }
    crosssell_map = {
        "gaming": [
            {"label": "+ Gaming Mouse", "msg": "What gaming mice do you recommend?"},
            {"label": "+ Gaming Headset", "msg": "Do you have gaming headset recommendations?"},
        ],
        "business": [
            {"label": "+ Docking Station", "msg": "What docking stations work with this laptop?"},
            {"label": "+ Extended Warranty", "msg": "Tell me about warranty options"},
        ],
        "creative": [
            {"label": "+ External Monitor", "msg": "What monitors pair well with this for creative work?"},
            {"label": "+ Stylus/Pen", "msg": "Do you recommend a stylus for creative work?"},
        ],
        "student": [
            {"label": "+ Laptop Bag", "msg": "What laptop bags do you recommend for students?"},
            {"label": "+ Student Discount", "msg": "Are there any student discounts available?"},
        ],
        "smb": [
            {"label": "+ Business Support", "msg": "What business support plans are available?"},
            {"label": "+ Bulk Pricing", "msg": "Do you offer bulk pricing for businesses?"},
        ],
    }
    cta_data["crosssell"] = crosssell_map.get(category or "all", [])
    return json.dumps(cta_data)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = session_manager.get_or_create(session_id)

    try:
        while True:
            user_msg = await websocket.receive_text()
            turn_start = time.time()
            session.turn_count += 1

            # ── LangGraph Decision ──
            state = graph.invoke({
                "session": session,
                "user_msg": user_msg,
                "action": None,
                "response": None,
                "category": None,
                "matched_products": None,
                "system_prompt": None,
                "skip": False,
            })

            action = state["action"]
            category = state.get("category") or "all"

            # ── Execute Action ──
            if action in ("direct", "clarify"):
                await websocket.send_text(state["response"])
                await websocket.send_text("[DONE]")
                if action == "clarify":
                    session.history.append(HumanMessage(content=user_msg))
                continue

            if action == "cache":
                await websocket.send_text(state["response"])
                cta = build_cta(category)
                await websocket.send_text(f"[CTA]{cta}")
                await websocket.send_text("[DONE]")
                latency = round(time.time() - turn_start, 2)
                await websocket.send_text(f"[METRICS] latency:{latency}s | route:{category} | type:cache | llm:skipped")
                session.history.append(HumanMessage(content=user_msg))
                continue

            if action == "factual":
                await websocket.send_text(state["response"])
                cta = build_cta(category)
                await websocket.send_text(f"[CTA]{cta}")
                await websocket.send_text("[DONE]")
                latency = round(time.time() - turn_start, 2)
                await websocket.send_text(f"[METRICS] latency:{latency}s | route:{category} | type:factual | llm:skipped")
                session.history.append(HumanMessage(content=user_msg))
                continue

            if action == "llm":
                # Streaming LLM call
                system_prompt = state["system_prompt"]
                context_history = session.history[-6:]
                messages = [SystemMessage(content=system_prompt)] + context_history + \
                           [HumanMessage(content=user_msg)]

                full_response = ""
                async for chunk in llm.astream(messages):
                    token = chunk.content
                    full_response += token
                    await websocket.send_text(token)

                latency = round(time.time() - turn_start, 2)
                token_estimate = len(system_prompt.split()) + len(user_msg.split()) + \
                                 len(full_response.split())

                cta = build_cta(category)
                await websocket.send_text(f"[CTA]{cta}")
                await websocket.send_text("[DONE]")
                await websocket.send_text(
                    f"[METRICS] latency:{latency}s | route:{category} | "
                    f"type:llm | tokens:~{token_estimate} | llm:called"
                )

                session.history.append(HumanMessage(content=user_msg))
                session.history.append(HumanMessage(content=full_response))

                # Store in cache
                session.cache.store_entry(user_msg, full_response, category)

    except WebSocketDisconnect:
        session_manager.clear(session_id)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)