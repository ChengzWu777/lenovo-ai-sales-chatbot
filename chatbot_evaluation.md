# AI Sales Chatbot — Evaluation Framework

**Author**: Chengzhen Wu
**Version**: 2.0

---

## Why This Document Exists

A system design without evaluation is a hypothesis. This document defines how to **measure** whether each step in the architecture works correctly, what "good" looks like, and what failure modes to watch for.

Core principle: **if you can't measure it, you can't improve it.**

This framework covers two evaluation scopes:
- **Phase 1 (MVP)**: Quick validation with 20 hand-written conversations. Goal: confirm core recommendation quality before adding optimizations.
- **Phase 2 (Full Architecture)**: Comprehensive evaluation with a 50-conversation golden dataset, per-step metrics, and regression detection. This document primarily covers Phase 2.

---

## Evaluation Strategy Overview

| Step | What to Evaluate | Method | When |
|------|-----------------|--------|------|
| 2. Cache | Hit rate, answer validity | Logging + spot-check | Runtime |
| 3. Context | Is 6 turns optimal? Profile accuracy? | A/B test, golden dataset | Pre-launch + runtime |
| 4. Intent | Classification accuracy | Golden dataset | Pre-launch |
| 5. Slots | Extraction accuracy and completeness | Golden dataset | Pre-launch |
| 6. Router | Category accuracy | Golden dataset | Pre-launch |
| 7. Match | Result relevance | Human review | Pre-launch |
| 8. LLM | Response quality | LLM-as-judge + human | Pre-launch + runtime |
| 10. Output | Conversion metrics | Analytics | Runtime |
| E2E | Full conversation quality | End-to-end test suite | Pre-launch |

Two types of evaluation:
- **Offline (pre-launch)**: Run against a prepared test dataset. Repeatable. Catches regressions.
- **Online (runtime)**: Monitor live traffic. Catches edge cases the test set missed.

---

## Golden Dataset Design

Before evaluating any step, we need a **golden dataset** — a set of test conversations with human-verified correct answers for each step.

### How to Build It

1. Write 50 realistic multi-turn conversations (5–8 turns each) covering:
   - Simple product inquiry (gaming laptop, budget clear)
   - Ambiguous query (no budget mentioned, vague use case)
   - Category edge case ("BizBook for video editing" — Business or Creative?)
   - Topic switch mid-conversation
   - Support/FAQ question mixed with product questions
   - Returning user asking similar question to past session
   - User changing mind ("actually forget gaming, show me business laptops")

2. For each turn in each conversation, manually label the **expected correct output** of every step:
   - Intent label (product / support / off-topic)
   - Extracted slots after this turn
   - Profile state after this turn
   - Router category
   - Expected product matches
   - Whether LLM should be called

3. Format as JSON:

```json
{
  "conversation_id": "test_001",
  "turns": [
    {
      "turn": 1,
      "user_message": "I need a laptop for video editing under $1500",
      "expected": {
        "intent": "product",
        "slots_after": {"use_case": "creative/video editing", "budget": "$1000-$1500"},
        "profile_after": {"use_case": "creative/video editing", "budget": "$1000-$1500"},
        "router_category": "Creative",
        "should_call_llm": false,
        "expected_products_contain": ["CreativeBook Pro", "CreativeBook Air"],
        "expected_products_exclude": ["GamingPro X15"]
      }
    }
  ]
}
```

**Size target**: 50 conversations × ~6 turns = ~300 labeled turns. Enough for meaningful accuracy numbers, small enough to label by hand.

---

## Step-by-Step Evaluation

### Step 2: Cache Layer

**What can go wrong**:
- False hit: returns a cached answer that doesn't match the current question
- Stale hit: returns an outdated answer (product price changed, product discontinued)
- Low hit rate: cache exists but rarely matches, providing no cost benefit

**Metrics**:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Cache hit rate | `hits / total_returning_user_queries` | >20% (baseline; improves over time) |
| False hit rate | Sample 100 cache hits, have human judge "did this answer the actual question?" | <5% |
| Staleness rate | Compare cached product data against current DB | 0% for price, <10% for specs |

**Evaluation method**:
- PoC: manual spot-check of every cache hit during testing
- Production: log every cache hit with the query and returned answer; weekly sample review

**Key design question**: "How do you prevent false cache hits?"
**Answer**: Keyword matching is conservative — it only returns a hit on high-overlap queries. Production embedding similarity adds a threshold (e.g., cosine >0.92). Log all hits for review.

---

### Step 3: Context Window — Profile Extraction Accuracy

**This is the most critical evaluation**, because every downstream step depends on Profile being correct.

**What can go wrong**:
- **Missed extraction**: user said "budget is $800" but Profile doesn't have budget
- **Wrong extraction**: user said "I don't want GamingLine" but Profile says `brand_preference: "GamingLine"`
- **Over-extraction**: user casually mentioned "my friend has a BizBook" but Profile sets `brand_preference: "BizBook"`
- **Stale data**: user changed mind but Profile still has old value

**Metrics**:

| Metric | Definition | How to Measure | Target |
|--------|-----------|---------------|--------|
| Slot extraction precision | Of all slots extracted, how many are correct? | Compare against golden dataset labels | >90% |
| Slot extraction recall | Of all slots that should be extracted, how many were? | Compare against golden dataset labels | >85% |
| Negation handling | When user says "NOT X", does Profile correctly record rejection, not preference? | Specific test cases in golden dataset | >95% |
| Profile consistency | After N turns, does Profile accurately reflect the full conversation? | Compare Profile snapshot at each turn against golden labels | >85% at turn 6+ |

**How to actually test this**:

```python
# Pseudo-code for profile evaluation
for conversation in golden_dataset:
    profile = {}
    for turn in conversation.turns:
        # Run your extraction logic
        profile = extract_and_update_profile(profile, turn.user_message, turn.assistant_response)
        
        # Compare against expected
        expected = turn.expected.profile_after
        
        for key in expected:
            if key not in profile:
                log_miss(conversation.id, turn.turn, key)  # Missed extraction
            elif profile[key] != expected[key]:
                log_wrong(conversation.id, turn.turn, key, profile[key], expected[key])  # Wrong value
        
        for key in profile:
            if key not in expected:
                log_over_extraction(conversation.id, turn.turn, key)  # Extracted something that shouldn't be there
```

**Critical test cases for Profile**:

| Input | Correct Extraction | Common Failure |
|-------|-------------------|----------------|
| "I don't want anything over 4.5 lbs" | `weight_max: "4.5 lbs"` | `weight_priority: "lightweight"` (too vague) |
| "I don't want GamingLine" | `brand_reject: ["GamingLine"]` | `brand_preference: "GamingLine"` (negation lost) |
| "Budget is firm at $800" | `budget: "$800", budget_firm: true` | `budget: "$600-$1000"` (range instead of firm) |
| "My friend has a BizBook but I'm open to anything" | No brand preference set | `brand_preference: "BizBook"` (over-extraction) |
| "Actually forget gaming, show me business laptops" | `use_case: "business"` (override gaming) | `use_case: "gaming"` still in profile (stale) |

**PoC evaluation approach**:
- If using rule-based extraction: test each rule against the critical test cases above
- If using LLM for extraction: run the golden dataset through, score precision/recall, iterate on the extraction prompt until targets are met
- Log every extraction at runtime for post-hoc review

**Why 6 turns? How to validate the threshold**:

Run the golden dataset at different thresholds (4, 6, 8, 10 turns) and measure:

| Threshold | Profile accuracy | LLM response quality | Input tokens per call |
|-----------|-----------------|---------------------|----------------------|
| 4 turns | Baseline | May miss recent context | Lowest cost |
| 6 turns | Expected: similar to full | Expected: close to full | Moderate cost |
| 8 turns | Expected: minimal gain | Expected: marginal gain | Higher cost |
| All turns | Ground truth | Ground truth | Highest cost |

If 6-turn accuracy is within 5% of full-history accuracy, the threshold is validated. If not, adjust.

---

### Step 4: Intent Classification

**What can go wrong**:
- Product question misclassified as support → user gets FAQ response instead of product recommendation
- Support question misclassified as product → bot asks "what's your budget?" in response to "return policy"

**Metrics**:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Classification accuracy | Golden dataset: predicted intent vs. labeled intent | >90% |
| False product rate | Support/off-topic questions that enter product pipeline | <5% |
| False support rate | Product questions routed to support | <3% |

**Confusion matrix** (from golden dataset):

```
                Predicted:
                Product  Support  Off-topic
Actual Product:   TP       FN       FN
Actual Support:   FP       TP       FN
Actual Off-top:   FP       FN       TP
```

**PoC**: run 300 golden turns through keyword rules, compute accuracy. If <90%, add more keywords or switch to LLM-based classification.

**Edge cases to include in golden dataset**:

| Input | Correct Label | Why It's Tricky |
|-------|--------------|-----------------|
| "Is the GamingPro X15 still available?" | Product | Could be mistaken for support (availability check) |
| "I'm having issues with my BizBook" | Support | Contains product name, could be mistaken for product inquiry |
| "What's the difference between your warranty plans?" | Support | "Difference between" sounds like product comparison |
| "Thanks, that's helpful" | Off-topic | Short positive feedback, not a question |

---

### Step 5: Slot Filling Accuracy

**What can go wrong**:
- Slot extracted from wrong part of message
- Slot not extracted even though user stated it clearly
- Slot schema too rigid — user says "around a thousand bucks" and system doesn't map to `budget: "$900-$1100"`

**Metrics**:

| Metric | Definition | Target |
|--------|-----------|--------|
| Per-slot precision | For each slot type, accuracy of extracted values | use_case >92%, budget >88% |
| Per-slot recall | For each slot type, rate of successful extraction when present | use_case >90%, budget >85% |
| Natural language handling | "around a grand", "a thousand bucks", "not too expensive" correctly mapped | >80% |

**Why budget targets are lower than use_case**: Budget expressions are more varied and ambiguous. "Not too expensive" could mean $500 or $1500 depending on context. Use case is usually more explicit.

**Test approach**:

```python
for turn in golden_dataset_turns:
    extracted_slots = extract_slots(turn.context, turn.user_message)
    expected_slots = turn.expected.slots_after
    
    for slot_name in ["use_case", "budget", "screen_size", "weight_priority", "brand_preference"]:
        expected_val = expected_slots.get(slot_name)
        extracted_val = extracted_slots.get(slot_name)
        
        if expected_val and not extracted_val:
            recall_miss(slot_name)
        elif extracted_val and not expected_val:
            precision_miss(slot_name)
        elif expected_val and extracted_val:
            if is_equivalent(expected_val, extracted_val):  # "$1000" == "$900-$1100" for "around a grand"
                correct(slot_name)
            else:
                wrong_value(slot_name)
```

**Note on equivalence**: Budget comparison needs fuzzy matching. "$1000" and "$900-$1100" are both valid extractions for "around a thousand." Define equivalence rules per slot type.

---

### Step 6: Query Router Accuracy

**What can go wrong**:
- Misclassification: "BizBook for video editing" → Business instead of Creative
- Ambiguous query: "lightweight laptop for work and occasional gaming" — Business or Gaming?
- Trigger word dominance: brand name overrides use case

**Metrics**:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Top-1 accuracy | Correct category on first try | >85% |
| Top-2 accuracy | Correct category in first or second choice | >95% |
| Fallback rate | How often router returns "unclear" | <15% |
| Misroute recovery | When first category yields 0 results, does reclassify succeed? | >70% |

**Why top-2 matters**: The reclassify mechanism (Step 7) gives us a second chance. If the correct category is in the top-2 85%+ of the time, the system works well even with imperfect first-pass routing.

**Critical test cases**:

| Input | Expected | Why Tricky |
|-------|----------|-----------|
| "BizBook for video editing" | Creative (not Business) | Brand vs. use case conflict |
| "Cheap gaming laptop" | Gaming + Budget (dual category) | Two categories apply |
| "Something for my kid going to college" | Budget or Business | No explicit use case keyword |
| "I need a workstation for 3D rendering" | Creative | "Workstation" could suggest Business |

---

### Step 7: Product Match Relevance

**What can go wrong**:
- Matched products don't fit the user's actual needs
- Budget filter too strict (user said "around $1000" but $1,099 product excluded)
- Budget filter too loose (showing $2000 products to a $1000 budget user)

**Metrics**:

| Metric | Definition | Target |
|--------|-----------|--------|
| Precision@3 | Of top 3 recommended products, how many are relevant? | >80% |
| Budget adherence | Recommended products within ±20% of stated budget | >90% |
| Constraint satisfaction | Products satisfy all hard constraints (weight, screen, etc.) | >95% |

**How to evaluate**:

For each test conversation that reaches Step 7, compare returned products against golden dataset's `expected_products_contain` and `expected_products_exclude`.

```python
for test in golden_dataset:
    if test.reaches_step_7:
        actual_products = run_product_match(test.slots)
        
        # Check inclusion
        for expected in test.expected_products_contain:
            if expected not in actual_products:
                log_missing_product(test.id, expected)
        
        # Check exclusion
        for excluded in test.expected_products_exclude:
            if excluded in actual_products:
                log_wrong_product(test.id, excluded)
        
        # Check budget
        for product in actual_products:
            if not within_budget(product.price, test.slots.budget, tolerance=0.2):
                log_budget_violation(test.id, product)
```

---

### Step 8: LLM Response Quality

**What can go wrong**:
- Hallucinated specs (says GamingPro X15 has RTX 4090 when it has RTX 4060)
- Recommendation contradicts user preferences (recommends heavy laptop despite weight concern)
- Response ignores Profile context
- Comparison is biased or incomplete

**Metrics**:

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Factual accuracy | Cross-check every spec/price in response against product DB | >95% |
| Profile adherence | Does response respect all Profile constraints? | >90% |
| Recommendation relevance | Human rating: 1-5 scale | Average >4.0 |
| Hallucination rate | Claims not supported by product DB data | <3% |

**Three evaluation methods, from cheapest to most rigorous**:

**Method 1 — Rule-based checks (automated, free)**:

```python
def check_factual_accuracy(llm_response, product_db):
    errors = []
    for product_name in extract_mentioned_products(llm_response):
        product = product_db.get(product_name)
        for claim in extract_claims(llm_response, product_name):
            # e.g., claim = "GamingPro X15 weighs 5.5 lbs"
            if not verify_against_db(claim, product):
                errors.append({"claim": claim, "product": product_name})
    return errors
```

**Method 2 — LLM-as-judge (automated, low cost)**:

Use a second LLM call to evaluate the first one's response:

```
System: You are a QA evaluator. Given the user's profile, product data, and the chatbot's response, 
rate the response on these criteria (1-5 each):
1. Factual accuracy: Are all specs and prices correct?
2. Profile adherence: Does it respect the user's stated constraints?
3. Helpfulness: Would this help the user make a decision?
4. Hallucination: Does it claim anything not in the product data?

Respond in JSON: {"accuracy": N, "adherence": N, "helpfulness": N, "hallucination": N}
```

Cost: ~$0.005 per evaluation (using Haiku). For 300 golden turns, total evaluation cost: ~$1.50.

**Method 3 — Human evaluation (manual, gold standard)**:

For critical test cases, have a human rate the response. Use for final validation and calibrating the LLM-as-judge.

---

### Step 10: Output — Business Metrics (Runtime Only)

These can only be measured with real users, not in a PoC. But defining them upfront ensures the system is designed with business outcomes in mind.

| Metric | Definition | How to Measure |
|--------|-----------|---------------|
| CTA click rate | % of recommendations where user clicks a CTA | Frontend event tracking |
| Cross-sell acceptance | % of cross-sell suggestions user engages with | Frontend event tracking |
| Conversation completion | % of conversations that reach a product recommendation | Session analysis |
| Avg turns to recommendation | How many turns before first product is shown | Session analysis |
| User satisfaction | Post-conversation rating (1-5) | Optional survey prompt |

**PoC**: Track turns-to-recommendation in the sample conversations. Target: ≤3 turns for clear queries, ≤5 for ambiguous ones.

---

### End-to-End Evaluation

Beyond per-step metrics, evaluate the full pipeline as a whole.

**Metric 1 — Path correctness**: For each golden conversation, does the system follow the expected path?

```python
for conversation in golden_dataset:
    session = create_session()
    for turn in conversation.turns:
        result = process_message(session, turn.user_message)
        
        # Check which steps were executed
        assert result.steps_executed == turn.expected.steps_executed
        
        # Check LLM was called only when expected
        assert result.llm_called == turn.expected.should_call_llm
        
        # Check final response is reasonable
        assert response_quality(result.response) > threshold
```

**Metric 2 — Cost efficiency** (Phase 2 targets — Phase 1 baseline is expected to be ~$0.06-$0.15 per conversation before optimizations):

| Metric | Target |
|--------|--------|
| LLM call rate across test set | 20-40% |
| Average tokens per conversation (6 turns) | <5,000 |
| Average cost per conversation | <$0.03 (Sonnet) |

**Metric 3 — Regression detection**: After any code change, re-run the full golden dataset. If any metric drops >2%, investigate before deploying.

---

## Evaluation Summary Table

| Step | Key Metric | Target | Method | Cost |
|------|-----------|--------|--------|------|
| 2. Cache | False hit rate | <5% | Human review of sampled hits | Free (manual) |
| 3. Profile | Extraction precision | >90% | Golden dataset comparison | Free (automated) |
| 3. Profile | Extraction recall | >85% | Golden dataset comparison | Free (automated) |
| 3. Profile | Negation handling | >95% | Specific test cases | Free (automated) |
| 4. Intent | Classification accuracy | >90% | Golden dataset | Free (automated) |
| 5. Slots | Per-slot precision | >88-92% | Golden dataset | Free (automated) |
| 6. Router | Top-1 accuracy | >85% | Golden dataset | Free (automated) |
| 6. Router | Top-2 accuracy | >95% | Golden dataset | Free (automated) |
| 7. Match | Precision@3 | >80% | Golden dataset | Free (automated) |
| 7. Match | Budget adherence | >90% | Golden dataset | Free (automated) |
| 8. LLM | Factual accuracy | >95% | Rule-based + LLM-as-judge | ~$1.50 total |
| 8. LLM | Hallucination rate | <3% | LLM-as-judge | ~$1.50 total |
| E2E | Path correctness | >90% | Golden dataset | Free (automated) |
| E2E | LLM call rate | 20-40% | Golden dataset | Free (automated) |

**Total evaluation cost**: ~$3 to evaluate the full golden dataset. This is negligible.

---

## What to Build for PoC vs. Production

| Component | PoC | Production |
|-----------|-----|-----------|
| Golden dataset | 50 conversations, hand-labeled | 500+, continuously expanded from real user data |
| Profile evaluation | Run golden set, check precision/recall | Automated pipeline after each deployment |
| Intent evaluation | Run golden set | A/B test keyword rules vs. classifier |
| LLM evaluation | LLM-as-judge on golden set | LLM-as-judge on sampled production traffic + human review |
| Cost tracking | Manual token counting | Automated per-request logging with dashboards |
| Regression tests | Run before each release | CI/CD pipeline blocks deployment on regression |

---

## Evaluation Philosophy

The core question this framework answers: **"How do you know this works?"**

1. **Golden dataset** — 50 conversations with human-labeled expected outputs for every step.
2. **Per-step evaluation** — profile extraction at >90% precision, intent classification at >90% accuracy, router at >85% top-1.
3. **End-to-end measurement** — path correctness, LLM call rate, cost per conversation.
4. **LLM-as-judge for response quality** — automated, costs ~$1.50 for the full test set, calibrated against human ratings.
5. **Regression detection** — re-run the golden set after every change, flag any metric drop >2%.

The goal is not just to build a system that works, but to build one where you can **prove** it works and **detect** when it stops working.

---

*Designed by Chengzhen Wu*
