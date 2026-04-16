Lenovo AI Sales Chatbot

An AI-powered sales assistant for Lenovo laptops, built as a personal project to demonstrate end-to-end AI application development. Features a multi-agent pipeline with LangGraph orchestration, RAG-based semantic product retrieval, and a local LLM for privacy-preserving inference.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple)

---

## Demo

The chatbot handles natural language queries across 60 Lenovo laptop SKUs (ThinkPad, ThinkBook, Legion, IdeaPad), with real-time streaming responses and context-aware recommendations.

**Example queries:**
- *"I need a gaming laptop under $1200, no AMD"*
- *"ThinkPad for business travel, must be under 3 lbs"*
- *"Something for video editing with color accurate display"*
- *"Compare Legion 5i vs Legion 7i"*

---

## Architecture
User Message
│
▼
┌─────────────────────────────────────────┐
│           LangGraph Pipeline            │
│                                         │
│  Intent → Cache → Slots → Clarify       │
│                              │          │
│                           Router        │
│                         (RAG + filter)  │
└─────────────────────────────────────────┘
│                    │
│ direct/cache/       │ llm
│ factual             ▼
│            ┌─────────────────┐
│            │  qwen3.5:27b    │
│            │  (Ollama local) │
│            │  astream()      │
│            └─────────────────┘
│                    │
▼                    ▼
WebSocket streaming → Chat UI

### Agent Modules

| Module | File | Purpose |
|--------|------|---------|
| Intent Classifier | `agents/intent_classifier.py` | Routes: product / support / greeting / off-topic |
| Slot Filler | `agents/slot_filler.py` | Extracts: budget, use_case, weight, processor, display |
| Query Router | `agents/query_router.py` | Category routing with profile filters |
| RAG Retriever | `agents/rag_retriever.py` | ChromaDB + MiniLM semantic search |
| Product Matcher | `agents/product_matcher.py` | Score + filter, factual shortcut |
| Cache Layer | `agents/cache_layer.py` | Keyword similarity cache, per-session |
| Session Manager | `agents/session_manager.py` | Per-connection profile + history |
| Sales Graph | `agents/sales_graph.py` | LangGraph DAG orchestration |

---

## Key Features

**Multi-agent LangGraph pipeline** — Each processing step (intent, cache, slots, routing, matching) is a separate graph node with conditional edges. Clean separation of decision logic from execution.

**RAG semantic retrieval** — Products indexed in ChromaDB using `sentence-transformers/all-MiniLM-L6-v2`. Handles semantic queries like *"ultralight laptop for frequent flyers"* that keyword matching would miss.

**Hard budget enforcement** — Slot filler extracts `budget_firm` flag. Products filtered before LLM sees them — the model cannot recommend what it cannot see.

**Factual shortcut** — Price/weight/spec queries bypass the LLM entirely (0.01s response) using direct DB lookup.

**Local LLM inference** — `qwen3.5:27b` via Ollama. Zero API cost, full data privacy. Streaming via `astream()` for real-time token output.

**CTA + Cross-sell** — Category-aware action buttons (View Specs / Compare / Talk to Rep) and cross-sell suggestions (Gaming → Mouse/Headset, Business → Dock/Warranty).

**Automated evaluation** — 40-test evaluation framework with budget compliance, keyword match, and category accuracy metrics.

---

## Evaluation Results

### Phase 1 → Phase 2 → Phase 3 (RAG)

| Metric | Phase 1 (Naive) | Phase 2 (Agents) | Phase 3 (RAG) |
|--------|----------------|-----------------|---------------|
| Budget compliance | 75% | 95% | **100%** |
| Avg latency | 84s | 49s | **40s** |
| Products to LLM | 60/60 | ~7/60 | ~10/60 (semantic) |
| Factual shortcut | ✗ | ✓ (0.01s) | ✓ (0.01s) |
| Semantic queries | ✗ | ✗ | ✓ |

### V1 Test Suite (20 cases)
Avg score:         0.96 / 1.00
Budget compliance: 20/20 (100%)
Keyword match:     20/20 (100%)
Category accuracy: 17/20 (85%)
Avg latency:       40.5s

---

## Tech Stack

- **Backend**: Python 3.14, FastAPI, WebSocket
- **Orchestration**: LangGraph
- **LLM**: qwen3.5:27b via Ollama (local inference)
- **RAG**: ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- **Frontend**: HTML/CSS/JS, WebSocket, marked.js
- **Hardware**: NVIDIA RTX 5090 32GB (local inference)

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- NVIDIA GPU recommended (RTX 3080+ for qwen3.5:27b)

### Installation

```bash
# Clone
git clone https://github.com/ChengzWu777/lenovo-ai-sales-chatbot.git
cd lenovo-ai-sales-chatbot

# Install dependencies
pip install fastapi uvicorn websockets langchain-ollama langgraph \
            chromadb sentence-transformers python-dotenv

# Pull LLM model
ollama pull qwen3.5:27b

# Index product catalog into ChromaDB
python -c "from agents.rag_retriever import get_retriever; get_retriever()"

# Start server
python main.py
```

Open `http://localhost:8000` in your browser.

### Run Evaluation

```bash
# V1 test suite (20 cases)
python evaluation/run_eval.py --suite v1

# V2 test suite (20 cases, edge cases + semantic)
python evaluation/run_eval.py --suite v2

# Full 40-case suite
python evaluation/run_eval.py --suite all
```

---

## Project Structure
ai-sales-chatbot/
├── main.py                    # FastAPI app, WebSocket handler, pipeline execution
├── agents/
│   ├── sales_graph.py         # LangGraph DAG — decision routing
│   ├── intent_classifier.py   # Intent classification
│   ├── slot_filler.py         # Structured slot extraction
│   ├── query_router.py        # Category routing + profile filters
│   ├── rag_retriever.py       # ChromaDB semantic retrieval
│   ├── product_matcher.py     # Product scoring + factual detection
│   ├── cache_layer.py         # Per-session keyword cache
│   └── session_manager.py     # Session profile + history
├── data/
│   └── products.json          # 60 Lenovo laptops (source of truth)
├── static/
│   └── index.html             # Chat UI + WebSocket client
└── evaluation/
├── test_cases.py          # 40 test cases (V1 + V2)
├── evaluator.py           # Automated evaluation engine
└── run_eval.py            # CLI evaluation runner

---

## Design Decisions

**Why local LLM?** Privacy, zero inference cost, and RTX 5090 provides sufficient performance for a PoC. Latency (~40s avg) is acceptable for a demo environment.

**Why LangGraph for routing only?** Initial implementation used LangGraph for all nodes including LLM. This killed streaming (140s vs 50s). Current architecture uses LangGraph for decision logic only, with `astream()` executed directly in the WebSocket handler.

**Why RAG over keyword routing?** Keyword routing fails on semantic queries like *"something for coffee shop work"* or *"laptop for nurses"*. RAG handles these naturally while maintaining fast retrieval.

**Phase-based development** — Each phase was evaluated against the same 20-query test set before proceeding. This mirrors production ML development practices where changes must be justified by measured improvements.

---

## Author

Chengzhen Wu — Senior Analyst Software Engineer  
[LinkedIn](https://linkedin.com/in/chengzhen-wu) | chriscwu588@gmail.com