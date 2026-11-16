# Changelog

## Review hardening

A code review of the initial version found correctness bugs in the deterministic pipeline and, more
importantly, that the evaluation could not have caught them. This release fixes both. Each item lists the
symptom, the cause, and the test that now pins it.

### Evaluation was measuring its own inputs

| Problem | Fix |
|---------|-----|
| The evaluator wrote `budget_max`, `budget_firm` and `weight_max` from each test case into the profile before running. "100% budget compliance" was the filter applied to the label. In the live pipeline "under $1000" was not firm and allowed $1,100. | Labels are assertions on slot extraction. Every case starts with an empty profile (`tests/test_evaluation.py::test_evaluator_does_not_inject_labels`). |
| Evaluation called the graph and the LLM directly: single turn, no history, temperature 0.3 vs 0.7 in production. | Evaluation calls `chat_service.run_turn`, the function the WebSocket handler uses. |
| `must_include` passed on any single keyword; a category of `all` always passed; `\$(\d{3,4})` missed `$1,299`. | Catalog-based constraint check, strict category check, price parser that handles commas. |
| Nothing checked whether the answer was grounded in the retrieved products. | Grounding check: named products must have been provided, quoted prices must match the catalog. |
| No multi-turn coverage. | 14 multi-turn conversations. |
| The evaluation design document described per-step metrics that were never implemented. | Slot, intent, routing and cache behaviour are covered by 196 unit tests; the document now has an implementation-status table. |

Running the rebuilt suite immediately found two further bugs (a message with only a dollar amount, and the
reply to a clarifying question, were both classified off-topic). Both are fixed and tested.

### Slot extraction (`agents/slot_filler.py`)

| Symptom | Cause | Now |
|---------|-------|-----|
| "I do NOT want AMD" recorded nothing; "I prefer Intel" recorded nothing | The preference / rejection block sat outside the vendor loop and only ran for the last vendor (`snapdragon`) | Evaluated per vendor; "Intel over AMD" prefers Intel without banning AMD |
| "keep me warm" → Snapdragon preference | Substring match on `arm` | Word-boundary matching everywhere |
| "budget laptop with 512GB SSD" → budget $512; "2-in-1" → budget range 2–1 | `budget.*?(\d+)` and the range pattern took any number | Spec numbers are stripped first; a number is a budget only in a money context; minimum $100 |
| "between $800 and $1200", "nothing over 4 lbs" not parsed; "under 1.5 kg" stored as 1.5 lbs | Missing patterns; unit ignored | Ranges, ceilings, floors, approximations, `k` suffix; kg → lbs |
| "must be under 3 lbs" set `budget_firm` | Firm keywords were matched on the whole message | Firmness is derived from how the budget itself is phrased |
| "video editing work" → business | First keyword match won; `work` is generic | Weighted scoring; negated mentions ("forget gaming") are ignored |
| New slots | | GPU floor, screen size, product-line preference, Chromebook exclusion, budget removal ("money is no object") |

Budget semantics changed: a stated budget is a hard ceiling. Only approximate phrasing ("around", "about",
"roughly") or "flexible" makes it soft (+10%).

### Cache (`agents/cache_layer.py`)

| Symptom | Cause | Now |
|---------|-------|-----|
| "gaming laptop under $2000" returned the cached "$1000" answer | Hit = 3 shared keywords; numbers were ordinary keywords; profile ignored | Key includes the profile signature and catalog version; numbers and negations must be identical; Jaccard ≥ 0.8 on the rest |
| A cache hit skipped slot extraction for that message | Cache node ran before the slot node | Order is intent → slots → cache |
| Unbounded growth | | Capped at 50 entries per session |

### Retrieval (`agents/filters.py`, `rag_retriever.py`, `product_matcher.py`, `sales_graph.py`)

- **Filter before ranking.** Previously the top 12 by similarity were fetched and then filtered, which could
  leave one or two products. Now the constraint filter runs on the catalog and ChromaDB ranks only the survivors.
- **One filter implementation.** The keyword route and the RAG route had separate, diverging filters (the
  keyword one had a `# Processor rejection` comment and no code). `query_router.py` is removed.
- **Semantic order is kept.** The heuristic re-sort used to discard the vector ranking; the two are now fused
  (reciprocal-rank fusion).
- **Relaxation is reported.** Weight / battery / display / form-factor filters were silently skipped when fewer
  than two products remained. Unmet soft constraints are now listed in a `NOTE:` in the prompt.
- **No-match path.** When no product satisfies the hard constraints the reply is deterministic and names the
  closest alternative. The old code reset any budget under $200 and tried again.
- **Named-model comparison.** `Intent.COMPARISON` was classified but never handled. Models named in the message
  are now resolved exactly ("Legion 5i vs Legion 7i"), with a comparison prompt.
- **Superlatives** ("cheapest", "lightest", "best battery", "most powerful") are exact sorts.
- **Follow-ups stay grounded.** "Compare these" / "the second one" keep the products from the previous answer.
- **Index versioning.** The index was only built when empty, so catalog edits were never picked up. It is now
  tagged with the catalog hash and embedding model and rebuilt on change. The catalog is cached in memory
  instead of re-read on every request.
- Gaming / workstation requests are no longer padded with office laptops.

### Intent (`agents/intent_classifier.py`)

- Unknown messages defaulted to a product inquiry, so "what's the weather", "write me a poem" and
  "ignore previous instructions ..." all reached the LLM. Without conversation context they are now off-topic;
  instruction-override patterns are always blocked.
- "hi" was off-topic (the length check ran before the greeting check).
- Word-boundary matching ("program" no longer contains "ram").
- New direct intents: accessory (the cross-sell buttons asked about products that are not in the catalog),
  human hand-off, thanks.

### Turn handling (`chat_service.py`, `main.py`)

- Assistant replies were appended to history as `HumanMessage`; direct, factual and cached replies were not
  recorded at all. Every turn now stores `HumanMessage` + `AIMessage`.
- The synchronous graph (embedding + vector search) ran on the event loop. It now runs in a worker thread.
- No timeouts. Added first-token and total deadlines; on timeout or connection failure the customer gets the
  top matches from the catalog and the turn is not cached.
- Any exception was printed and left the socket open with the UI locked. A failed turn now returns an error
  message and `[DONE]`; sessions are cleaned up in `finally`.
- Metrics now include time-to-first-token and tokens/s. `OLLAMA_REASONING` (default off) controls thinking tokens.
- Settings moved to environment variables; the server binds to `127.0.0.1` with reload off by default.
- The embedding model and index are loaded at startup.

### Frontend (`static/index.html`)

- Rendered markdown is sanitised with DOMPurify before insertion.
- The WebSocket URL is derived from the page location instead of `ws://localhost:8000`.

### Repository

- `requirements.txt`, `requirements-test.txt`, GitHub Actions workflow (unit tests + stub evaluation).
- Catalog count corrected: 65 products (README and metadata said 60).
- README rewritten; earlier evaluation numbers marked as retired with the reasons.

### After the first real-LLM run

- The grounding check compared a price against the wrong model when one catalog name is a prefix of another
  ("ThinkBook 14 Gen 7" / "ThinkBook 14 Gen 7 Intel"). Occurrences that belong to the longer name are now
  skipped (`tests/test_evaluation.py::test_check_grounding_does_not_confuse_sibling_models`).
- `evaluation/recheck.py` re-runs the text checks on a saved results file and prints the answer behind a case.
- README: real-LLM results and the reasoning on/off latency comparison (6.9 s vs 44.2 s average; time to first
  token 1.8 s vs 39.8 s).
