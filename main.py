import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agents.catalog import load_products
from agents.sales_graph import build_sales_graph
from agents.session_manager import session_manager
from chat_service import build_llm, run_turn
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chatbot")


def create_app(llm=None, graph=None, warm_up: bool = True) -> FastAPI:
    """App factory. Tests inject a fake LLM and a graph without the vector index."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:
            # Load the embedding model and index before the first customer message
            # instead of making that customer wait for it.
            try:
                from agents.rag_retriever import get_retriever
                await asyncio.to_thread(get_retriever)
            except Exception:
                log.exception("Vector index unavailable at startup; lexical ranking will be used")
        yield

    app = FastAPI(title="Lenovo AI Sales Chatbot (demo)", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.state.llm = llm if llm is not None else build_llm()
    app.state.graph = graph if graph is not None else build_sales_graph()

    @app.get("/")
    async def root():
        return FileResponse("static/index.html")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "products": len(load_products()), "model": settings.ollama_model}

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        await websocket.accept()
        session_id = str(uuid.uuid4())
        session = session_manager.get_or_create(session_id)
        try:
            while True:
                user_msg = (await websocket.receive_text()).strip()
                if not user_msg:
                    await websocket.send_text("[DONE]")
                    continue
                user_msg = user_msg[:settings.max_message_chars]
                try:
                    await run_turn(session, user_msg, app.state.graph, app.state.llm, websocket.send_text)
                except WebSocketDisconnect:
                    raise
                except Exception:
                    # One bad turn must not kill the conversation or leave the UI locked.
                    log.exception("Turn failed (session %s)", session_id)
                    await websocket.send_text("Sorry, something went wrong on my side. Please try again.")
                    await websocket.send_text("[DONE]")
        except WebSocketDisconnect:
            pass
        finally:
            session_manager.clear(session_id)

    return app


if __name__ == "__main__":
    uvicorn.run("main:create_app", factory=True, host=settings.host, port=settings.port,
                reload=settings.reload)
