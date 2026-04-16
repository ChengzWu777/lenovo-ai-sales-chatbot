# AI Sales Chatbot — System Design Document

**Author**: Chengzhen Wu
**Version**: 5.0

---

## Executive Summary

This document describes a two-phase approach to building a sales chatbot for a product recommendation workflow.

**Phase 1 (Core Workflow)**: Validate the fundamental loop — user message → LLM with product data → recommendation. Measure baseline quality, cost, and latency. Prove the science works before optimizing the engineering.

**Phase 2 (Full Architecture)**: Layer on optimizations one at a time, each solving a **measured problem** from Phase 1 — caching to reduce cost, query routing to improve precision, slot filling to structure extraction, graceful degradation to ensure reliability.

Core principle: **validate the core loop first. Optimize only what you've measured.**

---

## Pricing Reference

Source: official API pricing pages, verified March 2026.

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Haiku 4.5 | $1.00 | $5.00 |
| GPT-4o | $2.50 | $10.00 |
| GPT-4o Mini | $0.15 | $0.60 |

**Token estimates** (to be validated with actual tokenizer during development):

| Component | Estimated Tokens |
|-----------|-----------------|
| One turn (user + assistant) | ~200–400, midpoint 300 |
| System prompt | ~800 |
| Product data per item | ~50 |
| Profile (key-values) | ~200 |

---

## Phase 1: Core Workflow (MVP)

### The Simplest Possible Loop

```
User Message → LLM (system prompt + full history + product catalog) → Recommendation → Save
```

No cache. No router. No slot filling. Just: does the LLM give good product recommendations when given all available information?

### Why Start Here

Every optimization in Phase 2 is meaningless if the core recommendation quality is poor. Before adding complexity, we need answers to:

1. **Are LLM recommendations accurate?** Does it recommend appropriate products for the stated use case and budget?
2. **Does the LLM respect user preferences?** If the user says "nothing over 5 lbs", does the LLM comply?
3. **Is the conversation natural?** Does it feel like talking to a knowledgeable sales associate?
4. **What is the baseline cost?** How many tokens per conversation, at what dollar cost?
5. **What is the baseline latency?** How long does the user wait for each response?

### MVP Implementation

| Component | Choice |
|-----------|--------|
| Input | User message + full conversation history + entire product catalog (JSON) |
| LLM | Claude Sonnet 4.6 or GPT-4o (whichever API key available) |
| Product data | JSON file with ~25 products (name, price, specs, category) |
| System prompt | "You are a product advisor. Recommend products based on user needs. Always cite specific specs and prices from the product data provided." |
| Session | Python dict in memory |
| Frontend | Simple chat interface |

### MVP Evaluation

Run 20 test conversations (hand-written, covering simple and complex queries) for quick validation. Measure:

| Metric | How | Target to Move to Phase 2 |
|--------|-----|--------------------------|
| Recommendation accuracy | Human rates: did LLM recommend appropriate products? (1-5) | Average >3.5 |
| Preference adherence | Human checks: did LLM respect all stated constraints? | >85% |
| Factual accuracy | Cross-check specs/prices in response against product JSON | >90% |
| Avg tokens per conversation | Log actual token usage | Baseline (expect ~15,000-25,000 for 6 turns) |
| Avg cost per conversation | Tokens × pricing | Baseline (expect ~$0.06-$0.15 on Sonnet 4.6) |
| Avg latency per response | Measure response time | Baseline (expect 2-5s) |

### What This Baseline Tells Us

Once we have baseline numbers, we can identify **specific problems to solve**:

| Measured Problem | Solution (Phase 2 Step) |
|-----------------|------------------------|
| Cost too high: sending full catalog every turn wastes tokens | → Add Query Router (Step 6) to send only relevant products |
| Returning users repeat questions: redundant LLM calls | → Add Cache Layer (Step 2) to skip LLM on repeated queries |
| LLM asks irrelevant follow-ups for non-product questions | → Add Intent Classification (Step 4) to route support questions separately |
| LLM extracts preferences inconsistently | → Add Slot Filling (Step 5) with a defined schema |
| Long conversations blow up token budget | → Add Context Window management (Step 3) with Profile |
| LLM sometimes fails or hallucinates | → Add Graceful Degradation (Step 9) |
| Recommendations lack actionable next steps | → Add CTA + Cross-sell (Step 10) |

This is the evolution narrative: **each Phase 2 step exists because Phase 1 revealed a specific, measured problem.**

*Note: Phase 1 uses a quick 20-conversation test set for baseline validation. Phase 2 uses a comprehensive 50-conversation golden dataset with per-step labeling — see the companion Evaluation Framework document for full details.*

---

## Phase 2: Full Architecture

*Prerequisite: Phase 1 core workflow validated (recommendation accuracy >3.5, factual accuracy >90%).*

Each optimization layer below is annotated with the **problem it solves** and the **prerequisite measurement** from Phase 1.

### Architecture Overview

| Phase | Steps | Token Cost | Latency | Problem It Solves |
|-------|-------|-----------|---------|------------------|
| Entry | 1 | 0 | <10ms | — |
| Cache Layer | 2 | 0 (on hit) | <50ms | Redundant LLM calls for repeated questions |
| Context Window | 3 | 0 | <10ms | Token budget explosion in long conversations |
| Intent Classification | 4 | 0 | <10ms | Bad UX: support questions entering product flow |
| Slot Filling | 5 | 0 | <10ms | Inconsistent preference extraction |
| Query Router | 6 | 0 | <10ms | Sending full catalog wastes tokens, reduces precision |
| Product Match | 7 | 0 | <50ms | — |
| LLM + Degradation | 8–9 | ~3,500 tok (est.) | 2–5s | Need NL generation; need fallback when LLM fails |
| Output | 10 | 0 | <10ms | Recommendations lack conversion path |
| Save | 11 | 0 | <20ms | — |

---

### Step 1: User Sends Message

Raw text input. System assigns or retrieves `session_id`.

---

### Step 2: Cache Layer — Search Redis History

*Solves: redundant LLM calls for returning users asking similar questions.*

**What**: For returning users, search past QA pairs in Redis by keyword match.

**Outcomes**:
- **Hit** → return cached answer → Step 10 (Output) → Step 11 (Save). Zero LLM tokens.
- **Partial** → enrich current query with historical context → continue to Step 3.
- **Miss** → continue to Step 3. New users also go directly here.

**Cost per avoided call** (est. 3,000 input + 500 output tokens):
- Sonnet 4.6: ~$0.017 | GPT-4o: ~$0.013 | Haiku 4.5: ~$0.006
- Savings formula: `monthly_savings = daily_users × avg_turns × cache_hit_rate × cost_per_avoided_call × 30`

**PoC**: keyword matching. **Production**: embedding similarity (RediSearch, Pinecone).

---

### Step 3: Build Context Window

*Solves: token budget explosion in long conversations (measured in Phase 1 baseline).*

| Condition | Action |
|-----------|--------|
| 1–6 messages | Use ALL messages verbatim |
| 7+ messages | Last 6 messages verbatim + Profile |

**Profile** = structured key-value facts extracted from every prior turn. Updated at Step 11.

```
{use_case: "gaming", budget: "$800-$1000", screen_size: "15+", weight_priority: "lightweight"}
```

**Why 6 turns**: Cost-efficiency choice, not a context window constraint. 6 × 300 tok = 1,800 tok history. Keeps per-call input cost moderate. Threshold tunable based on Phase 1 data — if baseline shows average conversation is only 4 turns, 6 is generous. If average is 12, consider raising to 8.

**Why NOT summarize**: Summarization distorts hard constraints ("not over 4.5 lbs" → "prefers light"). Structured key-value extraction preserves exact facts.

---

### Step 4: Intent Classification

*Solves: support/FAQ questions ("return policy?") entering product pipeline and triggering irrelevant slot filling ("what's your budget?").*

| Intent | Action |
|--------|--------|
| Product inquiry | → Step 5 |
| Support / FAQ | → support flow → Step 11 |
| Off-topic | → redirect → Step 11 |

**PoC**: keyword rules. **Production**: lightweight classifier.

---

### Step 5: Slot Filling

*Solves: LLM extracting preferences inconsistently, making downstream product matching unreliable.*

| Slot | Required | Example |
|------|----------|---------|
| `use_case` | ✅ | gaming, business, creative |
| `budget` | ✅ | under $800, $1000–$1500 |
| `screen_size` | No | 15+, no preference |
| `weight_priority` | No | lightweight, not important |
| `brand_preference` | No | BizBook, GamingLine, none |

Both required → Step 6. Missing → ask targeted question → Step 11.

Slots accumulate in Profile across turns. User never repeats information.

---

### Step 6: Query Router (Intent-Based Routing)

*Solves: sending full product catalog (~5,000 tokens) to LLM every call, wasting tokens and reducing recommendation precision.*

**Note on terminology**: This is application-level query routing — classifying the user's query to select the relevant product subset. This is distinct from MoE (Mixture of Experts), which is a model-internal architecture where different parameter subsets activate for different inputs. The concept is analogous (conditional dispatch based on input type), but operates at a different abstraction layer. MoE happens inside the model weights; query routing happens in the application logic before the model is called.

| Category | Triggers |
|----------|----------|
| Gaming | gaming, FPS, GamingLine, streaming |
| Business | office, BizBook, enterprise |
| Creative | editing, design, CreativeBook, color accuracy |
| Budget | cheap, student, BudgetLine |
| Fallback | unclear → top 5 per category |

**Token impact**: ~100 products × 50 tok = 5,000 tok → after routing ~20 × 50 = 1,000 tok. ~80% reduction (structural ratio, not empirical).

**PoC**: keyword dictionary. **Production**: embedding vectors per category.

**Same-category follow-ups**: If user already routed to Gaming in a prior turn and asks about the same products, router result is reused from Profile.

---

### Step 7: Product Match

Search within routed category using filled slots (budget, constraints).

| Outcome | Action |
|---------|--------|
| Match (factual query) | → Step 10. No LLM. |
| Match (needs explanation/comparison) | → Step 8. |
| Zero results | → Reclassify to next category (max 1 retry) |

---

### Steps 8–9: Call LLM + Graceful Degradation

**Step 8**: LLM called only for subjective, comparative, or unresolvable queries.

Estimated prompt: system(~800) + profile(~200) + turns(~1,800) + filtered products(~200) = ~3,000 input + ~500 output = ~3,500 total tokens.

| Model | Est. Cost per Call |
|-------|--------------------|
| Sonnet 4.6 | ~$0.017 |
| GPT-4o | ~$0.013 |
| Haiku 4.5 | ~$0.006 |

**Step 9**: When LLM fails:

| Level | Trigger | Action |
|-------|---------|--------|
| L1 | Timeout / error | Show top 3 DB products, no explanation |
| L2 | Low confidence | Show products + "Want details?" |
| L3 | No products + fail | Transfer to human with context |

User always receives a response. Never a raw error.

---

### Step 10: Recommendations + CTA + Cross-sell

*Solves: recommendations without actionable next steps leave conversion on the table.*

**CTA**: View specs · Compare · Talk to rep
**Cross-sell**: Gaming → mouse, pad · Business → dock, case · Creative → stylus, monitor

PoC: hardcoded mappings. Production: collaborative filtering.

---

### Step 11: Save + Update Profile

Every turn: save conversation, extract facts to Profile, index QA for cache. All terminal paths converge here.

---

## Evolution Narrative

This is how the system grows, and why. Each step is justified by a measured gap:

```
Phase 1: MVP
  User → LLM (full catalog + full history) → Recommendation → Save
  
  Measured problems:
  ├── Cost too high (sending 5,000 tok of product data every call)
  ├── Returning users ask same questions (redundant LLM calls)  
  ├── Support questions enter product pipeline (bad UX)
  ├── Preferences extracted inconsistently
  ├── Long conversations blow up token budget
  └── No conversion path after recommendation

Phase 2: Full Architecture (each layer solves one problem)
  User → Cache check → Context window → Intent classification
       → Slot filling → Query router → Product match
       → LLM (only when needed, filtered products only)
       → CTA + Cross-sell → Save
```

This narrative makes the design defensible: each optimization layer addresses a specific measured problem from the baseline.

---

## Real Conversation Sample (6 Turns)

*Product names and prices are illustrative for PoC purposes.*

### Turn 1: "Hi, I'm looking for a new laptop"
- Cache: new user → skip
- Intent: product inquiry
- Slots: neither filled → ask "What will you use it for?"
- **LLM: No**

### Turn 2: "Mainly for gaming, FPS games"
- Slots: use_case ✅ gaming · budget ❌ → ask "Budget range?"
- **LLM: No**

### Turn 3: "Around $1000, maybe more if worth it"
- Slots: both filled ($1000–$1200)
- Router: Gaming
- Match: 3 products (GamingPro X15 $1,099, GamingLine S15 $999, BudgetGame V14 $749)
- Output: products + CTA + cross-sell
- **LLM: No** — factual data

### Turn 4: "GamingPro X15 seems heavy. How much does it weigh?"
- Router: reuse Gaming from Profile
- Match: DB lookup → 5.5 lbs. Also suggest GamingLine S15 (5.1 lbs).
- Profile updated: weight_priority = lightweight
- **LLM: No** — factual data

### Turn 5: "Compare the GamingLine S15 and GamingPro X15?"
- Match: both available → needs explanation → LLM
- LLM prompt: ~3,500 tokens (system + profile + 5 turns + 2 products)
- **LLM: Yes** — comparison requires natural language. Est. cost: ~$0.017 (Sonnet 4.6)

### Turn 6: "What's your return policy?"
- Intent: Support/FAQ → route to support flow
- **LLM: No** — FAQ

### Summary

| Metric | Value | Derivation |
|--------|-------|-----------|
| LLM calls | 1/6 (16.7%) | Only Turn 5 |
| Est. production LLM rate | 20–40% | Conservative: real traffic has more complex queries |
| Est. tokens (this sample, 1 call) | ~3,500 | Sum of prompt components |
| Tokens if naive (LLM every turn) | ~21,000 | 6 × 3,500 |
| Token reduction (this sample) | ~83% | (21,000 - 3,500) / 21,000 |
| Est. production token reduction | 60–80% | Adjusted for higher call rate |

*Token counts are estimates. Pricing from official pages, March 2026.*

---

## Assumptions & Verification

| Claim | Type | How to Verify |
|-------|------|--------------|
| Per-token pricing | Published fact | Official API pricing pages |
| ~300 tokens per turn | Estimate (midpoint) | Validate with tiktoken during dev |
| ~$0.017 per LLM call | Derived | Published price × estimated tokens |
| 80% routing reduction | Structural ratio | 100 products → 20 = 80% |
| 16.7% LLM call rate | Sample-specific | 1/6 in this sample. Production: 20-40% |
| 6-turn threshold | Design choice | Tunable. Validate with Phase 1 data. |

---

## Terminology Clarification

| Term Used | Meaning in This Document | NOT to Be Confused With |
|-----------|-------------------------|------------------------|
| Query Router | Application-level routing: classify user query to select product subset before DB search | MoE (Mixture of Experts): model-internal architecture where different parameter subsets activate per token |
| Profile | Structured key-value facts extracted from conversation, stored in session | User account or CRM profile |
| Slot Filling | Extracting predefined fields (use_case, budget) from conversation | Dialog state tracking in NLU frameworks (similar concept, simpler implementation) |
| Cache Layer | Storing past QA pairs for reuse | HTTP caching or CDN caching |
| Graceful Degradation | Fallback responses when LLM fails | Circuit breaker pattern (related but broader) |

---

## Future Iterations (Not in PoC)

| Feature | Priority | Solves |
|---------|----------|--------|
| Bilingual (EN/ZH) | High | APAC market |
| Embedding-based cache | High | Higher cache hit rate |
| Slot override | Medium | User changes mind |
| Tiered cache TTL | Medium | Stale pricing data |
| Metrics dashboard | Medium | Visibility into accuracy, cost, conversion |
| Multi-model routing (Haiku for simple, Sonnet for complex) | Low | Further cost optimization |

---

## Technical Stack (PoC)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backend | Python FastAPI | Async, lightweight |
| LLM | Claude API or OpenAI API | Whichever key available |
| Cache/Session | Python dict | PoC. Production: Redis |
| Product DB | JSON file (~25 products) | Production: PostgreSQL |
| Frontend | React or HTML/CSS | Lightweight, fast to prototype |
| Deployment | GitHub repo + README | Shareable, reproducible |

---

*Designed by Chengzhen Wu*
