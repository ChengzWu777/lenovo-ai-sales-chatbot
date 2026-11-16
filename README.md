# Lenovo AI Sales Chatbot

An AI sales assistant for Lenovo laptops, built as a personal project to show end-to-end AI application
engineering: a deterministic decision pipeline (LangGraph), constraint-first retrieval with semantic ranking
(ChromaDB), a locally hosted LLM (Ollama), and an evaluation suite that runs the production code path.

> Personal portfolio project. Not affiliated with or endorsed by Lenovo. Product data is a snapshot of
> public listings and may be out of date.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple)
![tests](https://github.com/ChengzWu777/lenovo-ai-sales-chatbot/actions/workflows/tests.yml/badge.svg)

---

## What it does

Natural-language shopping across 65 Lenovo laptop SKUs (ThinkPad, ThinkBook, Legion, IdeaPad), with streamed
answers and a customer profile that accumulates over the conversation.

- *"I need a gaming laptop under $1200, no AMD"*
- *"ThinkPad for business travel, must be under 3 lbs"*
- *"Something for video editing with a color accurate display"*
- *"Compare Legion 5i vs Legion 7i"*

---

## Architecture

```
User message
     │
     ▼
┌──────────────────── LangGraph (decides, never generates) ────────────────────┐
│                                                                              │
│  intent ──► slots ──► cache ──► clarify ──► retrieve                         │
│    │                    │          │           │                             │
│    │ greeting, support, │ same     │ vague     ├─ one named model + factual  │──► factual
│    │ off-topic,         │ question │ opener    ├─ nothing fits hard limits   │──► no_match
│    │ injection,         │ + same   │           ├─ "show me all ..."          │──► factual
│    │ accessory,         │ profile  │           └─ filter ► rank ► prompt     │──► llm
│    ▼ hand-off           ▼          ▼                                         │
│  direct               cache     clarify                                      │
└──────────────────────────────────────────────────────────────────────────────┘
     │
     ▼
chat_service.run_turn ── streams the LLM (first-token + total timeout, catalog fallback),
     │                   records history, emits CTA + metrics
     ▼
WebSocket ──► chat UI
```

Only the `llm` action calls the model. Everything else is answered deterministically from the catalog.

### Modules

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app factory, WebSocket endpoint, startup warm-up |
| `chat_service.py` | Runs one turn: graph → (stream LLM) → history, cache, metrics. Used by the server **and** the evaluator |
| `config.py` | Settings from environment variables |
| `agents/sales_graph.py` | LangGraph pipeline and the retrieve node |
| `agents/intent_classifier.py` | Rule-based intent, incl. injection / accessory / hand-off |
| `agents/slot_filler.py` | Budget, use case, processor, weight, GPU, display, battery, series |
| `agents/filters.py` | The single constraint filter (hard vs. soft, with relaxation reporting) |
| `agents/rag_retriever.py` | ChromaDB index + semantic ranking restricted to filtered candidates |
| `agents/product_matcher.py` | Rank fusion, superlatives, factual / list answers |
| `agents/catalog.py` | Cached catalog, content hash, exact model-name lookup |
| `agents/cache_layer.py` | Per-session answer cache keyed on question + profile + catalog version |
| `agents/taxonomy.py` | Shared use-case / category vocabulary |
| `agents/prompts.py` | Prompt templates and canned replies |
| `evaluation/` | Test cases, checks, evaluator, CLI |
| `tests/` | 195 unit / integration tests (no GPU, no model download) |

---

## Design decisions

**The LLM cannot break a constraint it never sees.** Hard constraints (budget ceiling, rejected processors,
rejected product lines) are applied in code before the prompt is built. A stated budget ("under $1000",
"budget is $800") is a ceiling; only approximate phrasing ("around $1200") gets a 10% tolerance.

**Filter first, then rank.** The constraint filter runs over the whole catalog, and the surviving ids are passed
to ChromaDB as a metadata filter, so the top-k comes from products the customer can actually buy. Ranking is a
reciprocal-rank fusion of the semantic order and a lexical / profile score.

**Soft constraints are relaxed out loud.** If nothing satisfies weight, battery, GPU, display or form-factor
requirements together, the unmet one is dropped *and* a `NOTE:` is added to the prompt telling the model to say
so. If nothing satisfies the hard constraints, the customer gets a direct "nothing fits, the closest is X" reply
with no LLM call.

**Exact questions get exact answers.** A price / weight / spec question about a named model, "show me all ...",
"cheapest", "lightest", "best battery" and named-model comparisons use lookups and sorts, not similarity search.

**LangGraph decides, the handler streams.** Running the LLM inside the graph prevented token streaming
(140 s vs 50 s in early experiments), so the graph returns an action and a prompt, and `run_turn` streams.

**Deterministic NLU.** Intent and slots are rules, not LLM calls: zero added latency, fully unit-testable, and
the failure modes are visible. The trade-off is coverage — see *Known limitations*.

**Local model.** `qwen3.5:27b` via Ollama on an RTX 5090: no per-token cost and no customer data leaving the
machine. Latency is the price; see *Latency* below.

**The cache is deliberately conservative.** A hit requires the same profile, the same catalog version, the same
numbers and negations, and near-identical keywords. A wrong cached answer is worse than a slow correct one.

---

## Evaluation

There are two layers.

### 1. Unit / integration tests — `pytest`

195 tests, about 3 seconds, no GPU and no model download (a deterministic embedder stands in for
sentence-transformers; ChromaDB itself is exercised for real). Covers slot extraction, intent, cache, model-name
lookup, constraint filtering, graph routing, turn handling (history, timeouts, fallback, cache), the WebSocket
endpoint, index rebuilds, and the evaluator's own checks.

```
pip install -r requirements-test.txt
pytest -q
```

### 2. End-to-end evaluation — `evaluation/run_eval.py`

54 cases: 40 single-turn queries plus 14 multi-turn conversations (budget stated late, change of mind,
cache false-hit, rejection carried across turns, follow-up comparison, withdrawn budget, impossible budget,
prompt injection ...). Every case starts with an **empty profile** and runs through `chat_service.run_turn`,
the function the WebSocket handler calls.

| Dimension | What is checked |
|-----------|-----------------|
| **slots** | The profile extracted from the conversation matches the labels |
| **routing** | Action, category, whether the LLM was called, relaxed-constraint notes |
| **constraints** | Every product shown to the model and every product named in the answer respects budget and exclusions — checked against the catalog, not by string matching |
| **grounding** | Products named in the answer were in the provided list; a price quoted next to a model equals the catalog price |
| **keywords** | `must_include` / `must_not_include` on the answer text |

```
python -m evaluation.run_eval --suite all                       # real LLM, needs Ollama + GPU
python -m evaluation.run_eval --suite all --llm stub            # pipeline + retrieval, no LLM
python -m evaluation.run_eval --suite all --llm stub --no-rag   # no vector index either (runs in CI)
```

`--llm stub` replaces the model with one that echoes the product list it was given. It cannot judge answer
quality, but it does answer "was the right product in front of the model, and were all constraints respected
before generation?"

### Results

**Deterministic pipeline (`--suite all --llm stub --no-rag`)** — reproducible anywhere:

| Metric | Result |
|--------|--------|
| Cases passed | 50 / 54 |
| Constraints | 54 / 54 |
| Slots | 54 / 54 |
| Routing | 50 / 54 |
| Keywords | 53 / 54 |
| Grounding | n/a (nothing is generated) |

The 4 failures are category labels that need semantic retrieval or are debatable (#22, #26, #33, #40) —
they are left failing rather than tuned away.

**Real LLM (`--suite all`)** — to be re-measured with the rebuilt evaluator:

| Metric | Result |
|--------|--------|
| Cases passed / Constraints / Grounding / Keywords / Slots / Routing | _run `python -m evaluation.run_eval --suite all`_ |
| LLM latency avg / p50 / p95, TTFT | _same run_ |

### Earlier results and why they were retired

The first evaluator (20 single-turn cases) reported 100% budget compliance, 0.96 average score and 40.5 s average
latency. Those numbers are **not comparable** with the tables above and should not be quoted as current:

- It wrote each case's `budget_max` / `budget_firm` labels into the profile before running, so "budget
  compliance" measured the filter against its own input. In the live pipeline "under $1000" was not treated as
  firm and allowed $1,100.
- `must_include` passed if any one keyword appeared, a category of `all` always passed, and the price regex
  missed `$1,299`.
- It ran single-turn, without history, at a different temperature than production.

The phase-over-phase story still holds directionally (sending ~10 filtered products instead of all 60 cut
latency roughly in half), but the absolute numbers need the re-run above. See [CHANGELOG.md](CHANGELOG.md).

---

## Latency

Each LLM turn reports `latency`, `ttft` (time to first visible token) and `tok/s` under the answer and in the
evaluation output. Things to check when optimising, in order of likely impact:

1. **Reasoning tokens.** Reasoning-capable models emit thinking tokens before the first visible token.
   `OLLAMA_REASONING` is `false` by default; set it to `true` and compare TTFT to see what thinking costs.
2. **Prompt size.** `MAX_PROMPT_PRODUCTS` (default 8) controls how many products are sent.
3. **Model size.** `OLLAMA_MODEL` — try a smaller model and compare the evaluation scores, not just the speed.
4. **Cold start.** The embedding model and index are loaded at startup, not on the first message.

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU recommended for `qwen3.5:27b`

### Install and run

```
git clone https://github.com/ChengzWu777/lenovo-ai-sales-chatbot.git
cd lenovo-ai-sales-chatbot
pip install -r requirements.txt
ollama pull qwen3.5:27b
python main.py
```

Open http://localhost:8000. The vector index is built on first start and rebuilt automatically whenever
`data/products.json` changes.

### Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_MODEL` | `qwen3.5:27b` | Model served by Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_REASONING` | `false` | Enable thinking tokens |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature (same for serving and evaluation) |
| `LLM_FIRST_TOKEN_TIMEOUT_S` | `60` | Give up and send the catalog fallback if no token arrives |
| `LLM_TOTAL_TIMEOUT_S` | `180` | Hard limit for one answer |
| `HISTORY_WINDOW` | `6` | Messages of history sent to the model |
| `MAX_PROMPT_PRODUCTS` | `8` | Products in the prompt |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `HOST` / `PORT` / `RELOAD` | `127.0.0.1` / `8000` / `false` | Server |

Windows `cmd`: `set OLLAMA_REASONING=true` then `python main.py`.

---

## Known limitations

- **Rule-based NLU has a coverage ceiling.** Unusual phrasings of budgets, negations or use cases will be missed.
  The next step is an LLM / small-classifier fallback when the rules extract nothing, measured against the
  existing slot tests.
- **Grounding checks are lexical.** They catch a model that names a product it was not given or misquotes a
  price, not subtle spec errors in free text. An LLM-as-judge pass is not implemented.
- **Category labels** feed only the cross-sell buttons; accuracy without the vector index is 50/54.
- **In-memory state.** Sessions and cache live in the process; no persistence, no multi-worker support.
- **Accessories, warranties and discounts are not in the catalog**, so those questions get an honest
  "not available in this demo" instead of a generated answer.
- **No live agent hand-off.** The hand-off reply shows the profile summary that would be passed on.
- **WebSocket framing is in-band** (`[DONE]`, `[CTA]`, `[METRICS]` text frames) rather than typed JSON messages.

---

## Project structure

```
lenovo-ai-sales-chatbot/
├── main.py                     # FastAPI app factory + WebSocket endpoint
├── chat_service.py             # one conversation turn (shared by server and evaluator)
├── config.py                   # environment-based settings
├── agents/
│   ├── sales_graph.py          # LangGraph pipeline
│   ├── intent_classifier.py
│   ├── slot_filler.py
│   ├── filters.py              # hard / soft constraints
│   ├── rag_retriever.py        # ChromaDB semantic ranking
│   ├── product_matcher.py      # rank fusion, superlatives, factual answers
│   ├── catalog.py              # catalog cache + model-name lookup
│   ├── cache_layer.py
│   ├── session_manager.py
│   ├── taxonomy.py
│   ├── prompts.py
│   └── text.py
├── data/products.json          # 65 laptops (source of truth)
├── static/index.html           # chat UI
├── evaluation/
│   ├── test_cases.py           # 40 single-turn + 14 multi-turn cases
│   ├── checks.py               # constraint / grounding / keyword checks
│   ├── evaluator.py
│   └── run_eval.py             # CLI
├── tests/                      # pytest suite
├── chatbot_design_document.md  # original design (pre-implementation)
├── chatbot_evaluation.md       # evaluation design + implementation status
└── CHANGELOG.md
```

---

## Author

Chengzhen Wu — Senior Analyst Software Engineer
[LinkedIn](https://linkedin.com/in/chengzhen-wu) | chriscwu588@gmail.com
