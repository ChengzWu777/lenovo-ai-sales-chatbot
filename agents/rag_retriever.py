"""Semantic ranking over the product catalog (ChromaDB + sentence-transformers).

The retriever only *orders* candidates. Constraint filtering happens first, in
agents/filters.py, and the surviving ids are passed in as a metadata filter, so
the top-k is taken from products the customer can actually buy. (Previously the
top-12 was fetched first and filtered afterwards, which could leave almost nothing.)
"""
import logging
import os
from typing import Callable, Optional

from agents.catalog import catalog_version, load_products
from config import settings

log = logging.getLogger(__name__)

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
COLLECTION = "lenovo_products"

EmbedFn = Callable[[list], list]


def product_document(p: dict) -> str:
    """Text that gets embedded. Describes what the laptop is for; prices are left
    out because '$999' carries no semantic signal for a sentence encoder."""
    return (
        f"{p['name']}. {p['series']} {p['category']} laptop. "
        f"Good for: {', '.join(p.get('use_cases', []))}. "
        f"Highlights: {', '.join(p.get('tags', []))}. "
        f"{p.get('description', '')} "
        f"Specs: {p['processor']}, {p['gpu']}, {p['ram']}, {p['display']}, "
        f"{p['weight_lbs']} lbs, battery {p['battery']}."
    )


def _default_embedder() -> EmbedFn:
    from sentence_transformers import SentenceTransformer  # heavy import, done lazily
    model = SentenceTransformer(settings.embed_model)
    return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()


class RAGRetriever:
    def __init__(self, embed_fn: Optional[EmbedFn] = None, path: str = CHROMA_PATH, client=None):
        import chromadb
        self.embed = embed_fn or _default_embedder()
        self.client = client or chromadb.PersistentClient(path=path)
        self.collection = None
        self.ensure_index()

    def _index_tag(self) -> str:
        return f"{catalog_version()}:{settings.embed_model}"

    def ensure_index(self):
        """(Re)build the index when the catalog file or the embedding model changes."""
        tag = self._index_tag()
        collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine", "index_tag": tag}
        )
        stale = (collection.metadata or {}).get("index_tag") != tag
        if stale or collection.count() != len(load_products()):
            self.client.delete_collection(COLLECTION)
            collection = self.client.create_collection(
                name=COLLECTION, metadata={"hnsw:space": "cosine", "index_tag": tag}
            )
            self._index_products(collection)
        self.collection = collection
        self._tag = tag

    def _index_products(self, collection):
        products = load_products()
        docs = [product_document(p) for p in products]
        collection.add(
            ids=[p["id"] for p in products],
            documents=docs,
            embeddings=self.embed(docs),
            metadatas=[{"pid": p["id"], "series": p["series"], "category": p["category"],
                        "price": p["price"]} for p in products],
        )
        log.info("RAG: indexed %d products (tag %s)", len(docs), collection.metadata.get("index_tag"))

    def rank(self, query: str, candidate_ids: Optional[list] = None, n_results: int = 12,
             use_case: Optional[str] = None) -> list:
        """Product ids ordered by semantic similarity, restricted to candidate_ids."""
        if self._tag != self._index_tag():
            self.ensure_index()
        total = self.collection.count()
        if total == 0 or candidate_ids == []:
            return []
        text = f"{query} ({use_case} use)" if use_case else query
        kwargs = {"query_embeddings": self.embed([text]), "n_results": min(n_results, total)}
        if candidate_ids is not None and len(candidate_ids) < total:
            kwargs["where"] = {"pid": {"$in": list(candidate_ids)}}
            kwargs["n_results"] = min(n_results, len(candidate_ids))
        results = self.collection.query(**kwargs)
        return results["ids"][0]


_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
