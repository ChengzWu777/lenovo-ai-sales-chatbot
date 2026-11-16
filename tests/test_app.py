from fastapi.testclient import TestClient

from agents.sales_graph import build_sales_graph
from agents.session_manager import session_manager
from main import create_app
from tests.conftest import FakeLLM


def make_client(llm=None, graph=None):
    app = create_app(llm=llm or FakeLLM(), graph=graph or build_sales_graph(retriever=None), warm_up=False)
    return TestClient(app)


def read_turn(ws):
    frames = []
    while True:
        frame = ws.receive_text()
        frames.append(frame)
        if frame == "[DONE]":
            return frames


def test_healthz_and_index():
    with make_client() as client:
        body = client.get("/healthz").json()
        assert body["status"] == "ok" and body["products"] == 65
        html = client.get("/").text
        assert "DOMPurify.sanitize" in html and "ws://localhost" not in html


def test_websocket_conversation_and_session_cleanup():
    before = len(session_manager.sessions)
    with make_client() as client:
        with client.websocket_connect("/ws/chat") as ws:
            assert len(session_manager.sessions) == before + 1
            ws.send_text("hi")
            assert "Sales Assistant" in read_turn(ws)[0]
            ws.send_text("I need a gaming laptop under $1000")
            frames = read_turn(ws)
            assert frames[-2].startswith("[CTA]")
            assert ws.receive_text().startswith("[METRICS]")
            ws.send_text("   ")
            assert read_turn(ws) == ["[DONE]"]
    assert len(session_manager.sessions) == before


def test_a_failing_turn_does_not_kill_the_connection():
    # Was: any exception was printed and the socket was left hanging with the UI locked.
    class ExplodingGraph:
        calls = 0

        def invoke(self, state):
            ExplodingGraph.calls += 1
            if ExplodingGraph.calls == 1:
                raise RuntimeError("boom")
            return build_sales_graph(retriever=None).invoke(state)

    before = len(session_manager.sessions)
    with make_client(graph=ExplodingGraph()) as client:
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("gaming laptop under $1000")
            frames = read_turn(ws)
            assert "something went wrong" in frames[0]
            ws.send_text("hi")
            assert "Sales Assistant" in read_turn(ws)[0]
    assert len(session_manager.sessions) == before
