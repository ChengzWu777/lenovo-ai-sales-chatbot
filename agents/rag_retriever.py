import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from agents.session_manager import UserProfile

PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.json")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

class RAGRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="lenovo_products",
            metadata={"hnsw:space": "cosine"}
        )
        if self.collection.count() == 0:
            self._index_products()

    def _index_products(self):
        with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        products = data["products"]

        docs, ids, metadatas = [], [], []
        for p in products:
            text = (
                f"{p['name']} {p['series']} {p['category']} "
                f"${p['price']} {p['processor']} {p['gpu']} "
                f"{p['ram']} {p['display']} {p['weight_lbs']}lbs "
                f"battery {p['battery']} "
                f"use cases: {' '.join(p.get('use_cases', []))} "
                f"tags: {' '.join(p.get('tags', []))} "
                f"{p.get('description', '')}"
            )
            docs.append(text)
            ids.append(p["id"])
            metadatas.append({
                "name": p["name"],
                "series": p["series"],
                "category": p["category"],
                "price": p["price"],
                "weight_lbs": p["weight_lbs"],
                "processor": p["processor"],
                "gpu": p["gpu"],
            })

        embeddings = self.model.encode(docs).tolist()
        self.collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"RAG: indexed {len(docs)} products into ChromaDB")

    def retrieve(self, query: str, profile: UserProfile, n_results: int = 10) -> list:
        # Build enriched query from profile
        enriched_query = query
        if profile.use_case:
            enriched_query += f" {profile.use_case}"
        if profile.budget_max:
            enriched_query += f" under ${profile.budget_max}"
        if profile.processor_pref:
            enriched_query += f" {profile.processor_pref} processor"
        if profile.weight_max:
            enriched_query += f" under {profile.weight_max} lbs"

        query_embedding = self.model.encode([enriched_query]).tolist()

        where_filter = self._build_filter(profile)

        try:
            if where_filter:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(n_results, self.collection.count()),
                    where=where_filter
                )
            else:
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=min(n_results, self.collection.count()),
                )
        except Exception:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(n_results, self.collection.count()),
            )

        # Load full product objects
        retrieved_ids = results["ids"][0]
        with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
            all_products = json.load(f)["products"]

        id_to_product = {p["id"]: p for p in all_products}
        products = [id_to_product[pid] for pid in retrieved_ids if pid in id_to_product]

        # Apply hard filters
        products = self._apply_hard_filters(products, profile)

        return products

    def _build_filter(self, profile: UserProfile) -> dict:
        conditions = []

        if profile.budget_max and profile.budget_firm:
            conditions.append({"price": {"$lte": profile.budget_max}})

        if profile.weight_max:
            conditions.append({"weight_lbs": {"$lte": profile.weight_max}})

        if len(conditions) == 0:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _apply_hard_filters(self, products: list, profile: UserProfile) -> list:
        result = products

        if profile.budget_max:
            if profile.budget_firm:
                result = [p for p in result if p["price"] <= profile.budget_max]
            else:
                result = [p for p in result if p["price"] <= profile.budget_max * 1.1]

        if profile.weight_max:
            filtered = [p for p in result if p.get("weight_lbs", 99) <= profile.weight_max]
            if len(filtered) >= 2:
                result = filtered

        for rejected in profile.processor_reject:
            result = [p for p in result if rejected.lower() not in p.get("processor", "").lower()]

        if profile.processor_pref:
            pref = profile.processor_pref.lower()
            preferred = [p for p in result if pref in p.get("processor", "").lower()]
            others = [p for p in result if pref not in p.get("processor", "").lower()]
            result = preferred + others

        return result

# Singleton
_retriever = None

def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever