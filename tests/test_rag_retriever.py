"""Exercises the real ChromaDB calls with a deterministic bag-of-words embedder,
so no model download or GPU is needed."""
import hashlib
import math
import shutil

import pytest

chromadb = pytest.importorskip("chromadb")

from agents import catalog, rag_retriever  # noqa: E402
from agents.catalog import load_products  # noqa: E402
from agents.rag_retriever import RAGRetriever  # noqa: E402
from agents.text import tokenize  # noqa: E402

DIM = 256


def fake_embed(texts):
    vectors = []
    for text in texts:
        v = [0.0] * DIM
        for token in tokenize(text):
            v[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors.append([x / norm for x in v])
    return vectors


@pytest.fixture
def retriever(tmp_path):
    return RAGRetriever(embed_fn=fake_embed, path=str(tmp_path / "chroma"))


def test_index_contains_every_product(retriever):
    assert retriever.collection.count() == len(load_products()) == 65


def test_rank_is_restricted_to_candidates(retriever):
    cheap = [p["id"] for p in load_products() if p["price"] <= 600]
    ranked = retriever.rank("laptop for school homework", candidate_ids=cheap, n_results=5, use_case="student")
    assert len(ranked) == 5 and set(ranked) <= set(cheap)


def test_rank_without_candidates_searches_everything(retriever):
    ranked = retriever.rank("legion gaming rtx 4090 flagship", n_results=3)
    assert "LG-9I-G9" in ranked


def test_rank_with_empty_candidate_list_returns_nothing(retriever):
    assert retriever.rank("anything", candidate_ids=[], n_results=5) == []


def test_small_candidate_set_does_not_over_request(retriever):
    two = [p["id"] for p in load_products()][:2]
    assert sorted(retriever.rank("laptop", candidate_ids=two, n_results=12)) == sorted(two)


def test_index_is_rebuilt_when_catalog_changes(tmp_path, monkeypatch):
    # Was: the index was only built when empty, so catalog edits were never picked up.
    path = str(tmp_path / "chroma")
    first = RAGRetriever(embed_fn=fake_embed, path=path)
    old_tag = first.collection.metadata["index_tag"]

    copy = tmp_path / "products.json"
    shutil.copy(catalog.PRODUCTS_PATH, copy)
    text = copy.read_text(encoding="utf-8").replace('"price": 299,', '"price": 289,', 1)
    assert text != copy.read_text(encoding="utf-8")
    copy.write_text(text, encoding="utf-8")
    monkeypatch.setattr(catalog, "PRODUCTS_PATH", str(copy))
    monkeypatch.setitem(catalog._cache, "mtime", None)

    second = RAGRetriever(embed_fn=fake_embed, path=path, client=first.client)
    assert second.collection.metadata["index_tag"] != old_tag
    assert second.collection.count() == 65
    assert min(p["price"] for p in load_products()) == 289
    monkeypatch.undo()
    catalog._cache["mtime"] = None
    assert min(p["price"] for p in load_products()) == 299


def test_graph_with_real_chroma_retriever(retriever):
    from agents.sales_graph import build_sales_graph, new_state
    from agents.session_manager import Session
    graph = build_sales_graph(retriever=retriever)
    state = graph.invoke(new_state(Session("x"), "ultralight laptop for frequent flyers under $1500"))
    assert state["action"] == "llm" and state["matched_products"]
    assert all(p["price"] <= 1500 for p in state["matched_products"])
    assert rag_retriever.product_document(state["matched_products"][0])
