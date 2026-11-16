import pytest

from agents.sales_graph import build_sales_graph, new_state
from agents.session_manager import Session


class FakeChunk:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class FakeLLM:
    """Stands in for ChatOllama: records the messages and streams a fixed answer."""

    def __init__(self, answer="Here is my recommendation: the best match from the list above. " * 3,
                 delay=0.0, fail=False, leading_empty=0):
        self.answer, self.delay, self.fail, self.leading_empty = answer, delay, fail, leading_empty
        self.calls = []

    async def astream(self, messages):
        import asyncio
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("ollama not reachable")
        for _ in range(self.leading_empty):
            yield FakeChunk("")
        for word in self.answer.split(" "):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield FakeChunk(word + " ")


@pytest.fixture(scope="session")
def graph():
    # No vector index in unit tests: lexical ranking only.
    return build_sales_graph(retriever=None)


@pytest.fixture
def session():
    return Session("test")


@pytest.fixture
def decide(graph):
    """Run one message through the decision graph and return the final state."""
    def _decide(session, message):
        session.turn_count += 1
        state = graph.invoke(new_state(session, message))
        if state.get("matched_products"):
            session.last_product_ids = [p["id"] for p in state["matched_products"]]
        return state
    return _decide
