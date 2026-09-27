# Failure & Limitation Matrix

**Purpose**: Honest documentation of system limitations and known failure modes for technical paper and defense.

**Context**: pasa_realscholar_test_b3b570411ce2399c (22 queries, 184 Gold)

---

## 1. Retrieval Coverage Limitation

### Problem
Large fraction of Gold papers never enter candidate pool, even after two-round retrieval.

### Metrics
- FAST: final_unique_gold = 14/184 (7.6% Gold coverage)
- DEEP: final_unique_gold = 15/184 (8.2% Gold coverage)
- Round-2 incremental: +2 Gold in retrieval pool, +1 in final output

### Root Causes
1. **Query-paper term mismatch**: Generated queries use different terminology than Gold paper titles/abstracts
2. **OpenAlex coverage**: Some Gold papers may not be indexed or have incomplete metadata
3. **Lexical retrieval limits**: BM25-style retrieval misses semantic matches with low lexical overlap
4. **Entity/method term variation**: Multiple names for same concepts (e.g., "LLM" vs "large language model" vs specific model names)

### Implications
- Current system achieves **precision-focused** results (low false positive in top-k)
- **Recall remains primary challenge** (many Gold papers not retrieved)
- Ranking quality secondary to retrieval coverage

### Not a Ranking Failure
- Given current candidate pool, reranker achieves highest observed F1
- The fundamental bottleneck is **getting Gold into the pool**, not ranking it once present

---

## 2. Iterative Search Marginal Gain Limitation

### Problem
Evidence-guided two-round retrieval has limited cost-benefit at scale.

### Metrics
- Round-2 executed queries: 62
- Incremental Gold (retrieval pool): +2 (25 → 27)
- Incremental Gold (final output): +1 (14 → 15)
- Gold/Query ratio: 2/62 ≈ 0.032
- F1 change: FAST 0.0680 vs DEEP 0.0641 (negative)

### Root Causes
1. **Full-deployment cost**: All 22 queries trigger Round-2, not just complex queries
2. **Generic follow-ups**: Some generated queries too broad or duplicate Round-1 coverage
3. **Reranking pool saturation**: Newly discovered papers compete with existing high-quality candidates
4. **Entity extraction noise**: Not all Round-1 evidence yields productive follow-ups

### Design Decision
- DEEP mode provides **capability demonstration** and **optional deep search**
- FAST mode is **default** for production (better cost-benefit)
- Selective Round-2 triggering (query complexity gating) not implemented

### Implications
- Mechanism validated: evidence-guided search CAN discover out-of-Round-1 papers
- Scale economics unfavorable: full-deployment marginal gain < cost
- Future work: smarter Round-2 trigger conditions (not "always on")

---

## 3. LLM Reranker Nondeterminism

### Problem
LLM-based reranker introduces stochasticity even with temperature=0.0.

### Evidence
- Multiple runs on same candidate pool yield slightly different rankings
- Top-k results generally stable, but boundary cases (rank 9-11) can shift
- Explanation text varies across runs even for same paper

### Implications
**Quality**: Current highest F1 in evaluation

**Stability**: Worse than deterministic rerankers (RRF, CE, BGE)

**Reproducibility**: Frozen plans + cache enable offline replay, but online re-execution may differ

### Tradeoff
System prioritizes **quality** (LLM achieves 0.0664 vs deterministic 0.0588-0.0598) over **perfect determinism**.

Deterministic rerankers available as fallback for stability-critical deployments.

### Mitigation
- Frozen plans ensure offline reproducibility
- SearchTrace records actual execution
- Temperature=0.0 reduces (doesn't eliminate) variance
- Cache replay guarantees exact reproduction for evaluation

---

## 4. Evaluation Scale Limitation

### Problem
22-query frozen subset is small for statistical generalization.

### Metrics
- Frozen queries: 22
- Total available: 50
- Frozen Gold: 184
- Full corpus Gold: 791

### Implications
- Results demonstrate **system capability** on validated queries
- Do NOT claim: "production performance on general academic search"
- Do NOT claim: "statistically significant over all query types"
- Correct framing: "Evaluated on 22-query frozen subset (184 Gold papers) from PaSa RealScholarQuery benchmark"

### Why 22-Query Subset
- Inherited from historical M3-R/M5A experiment scope
- Enables controlled ablation with consistent query set
- Frozen plans enable offline reproducibility without API cost

### Limitation Disclosure Required
Technical paper must state evaluation scope explicitly, not generalize to full user market.

---

## 5. Online Acceptance Pending

### Problem
Online validation (real Planner + OpenAlex calls) remains PENDING due to LLM API balance.

### Status
```
S1_OFFLINE_ACCEPTANCE = PASS
S1_ONLINE_ACCEPTANCE = PENDING_LLM_BALANCE
```

### What IS Validated
- Offline replay: 22/22 queries, api_calls=0 (cache hit 100%)
- Budget enforcement: 62/66 executed ≤ configured max
- Gold leakage: 0 violations (static + runtime tests)
- Structured output: 0 empty titles
- SearchTrace: complete execution logs

### What Is NOT Validated
- Real-time Planner invocation (online)
- Real-time OpenAlex API calls (online)
- Real-time Reranker invocation (online)
- Production latency measurement
- End-to-end online smoke tests

### Implications
- **Algorithmic correctness validated** via offline replay
- **Production deployment readiness** not fully validated
- External execution blocker (LLM balance), not system failure
- Deferred items documented in `eval/runs/s1/S1_DEFERRED_ACCEPTANCE.md`

### Required Disclosure
- Papers/defense must state: "Offline evaluation validated; online acceptance pending infrastructure availability"
- Do NOT claim: "Production-ready online system fully validated"
- Do NOT hide PENDING status

---

## 6. Production Cost Measurement Gap

### Problem
Offline replay (api_calls=0) does not reflect production cost.

### What IS Measured
- LLM calls: 3.4/query (FAST), 5.0/query (DEEP)
- Tokens: ~7930/query (FAST), ~11834/query (DEEP)
- Offline latency: ~14s/query (FAST), ~22s/query (DEEP)

### What Is NOT Measured
- Logical OpenAlex API calls per query (estimated from frozen plans, not measured end-to-end)
- Production latency including real API round-trips
- Cache miss behavior in production

### Artifacts
```
production_logical_api_calls_per_query = NOT_MEASURED
production_latency_estimate = NOT_MEASURED
```

### Implications
- Efficiency report distinguishes: **replay cost** (actual=0 HTTP) vs **production-equivalent logical cost** (estimated)
- Technical paper must NOT present api_calls=0 as production efficiency
- Correct framing: "Offline validation uses cached responses; production deployment requires OpenAlex API calls"

---

## 7. Metadata Source Limitation

### Problem
Factual metadata quality depends entirely on OpenAlex data completeness.

### Design Policy
- Title, authors, year, venue, DOI, OpenAlex ID: MUST come from academic source
- LLM forbidden from generating factual metadata
- LLM only generates: relevance_explanation, relevance_score, relevance_label

### Implications
**Correctness**: Metadata traceable to authoritative source

**Completeness**: Limited by OpenAlex coverage
- Some papers missing DOI
- Some papers incomplete author lists
- Some papers missing venue

### Observed
- S1 validation: 0 empty titles (all papers have title from OpenAlex)
- Per-field completeness NOT measured in frozen artifacts

### Tradeoff
System prioritizes **correctness** (source-attributed metadata) over **completeness** (LLM gap-filling).

---

## 8. Query Planning Not Adaptive

### Problem
Query planner uses frozen M3-R plans; does not adapt to retrieval results within single query.

### Evidence
- Frozen plans generated once per query
- No dynamic query reformulation based on initial retrieval quality
- Round-2 (DEEP) uses frozen M5A follow-ups, not dynamic planning

### Implications
**Reproducibility**: Perfect (frozen plans)

**Adaptivity**: Limited (no closed-loop feedback within single query)

### Design Decision
Prioritizes **reproducibility** and **budget control** over **adaptive exploration**.

Future: online adaptive planning would require careful budget management to avoid unbounded cost.

---

## 9. Deterministic Reranker Quality Gap

### Problem
Deterministic rerankers (CE, BGE, RRF) achieve lower F1 than LLM in current evaluation.

### Metrics
- LLM: 0.0664
- CE (MiniLM): 0.0598 (-10%)
- BGE v2-m3: 0.0588 (-11%)
- RRF: 0.0241 (-64%)

### Analysis
**Not a deterministic method failure**, but a **current config/pool** result.

Deterministic rerankers have advantages:
- Zero stochasticity
- Faster inference
- No LLM API cost
- Some systematic cases better than LLM

### Limitation
Current system uses LLM reranker for **quality**, accepts **stochasticity cost**.

Alternative: hybrid (deterministic primary, LLM refinement) not explored.

---

## 10. SearchTrace Completeness

### Problem
Frozen S1 artifacts have partial SearchTrace (no per-paper provenance).

### What IS Traced
- Original question
- Planner version / prompt hash
- Generated queries
- API calls / LLM calls / tokens / latency
- Round-1 observation
- Round-2 decision (DEEP)
- Final paper canonical IDs

### What Is NOT Traced in Frozen Artifacts
- Per-paper retrieval source (which query returned this paper)
- Per-paper reranker score evolution
- Candidate pool composition before reranking

### Implications
- High-level execution auditable
- Fine-grained per-paper provenance requires enhanced SearchTrace schema
- Current: query-level trace
- Future: paper-level trace

---

## Limitation Disclosure Checklist

For technical paper / defense, MUST disclose:

- [x] Recall primary bottleneck (not ranking)
- [x] Iterative search limited marginal gain at scale
- [x] LLM reranker nondeterminism
- [x] 22-query evaluation scope (not general market)
- [x] Online acceptance PENDING
- [x] Production cost NOT_MEASURED
- [x] Metadata completeness depends on OpenAlex
- [x] Query planning not adaptive within single query
- [x] Deterministic reranker quality gap
- [x] SearchTrace partial (no per-paper provenance)

**Do NOT**:
- Claim production-ready without online validation
- Present api_calls=0 as production efficiency
- Generalize 22-query results to all academic search
- Hide LLM stochasticity
- Claim Round-2 always improves F1
- Oversell deterministic reranker results

**Honest framing**: System demonstrates query planning, preserve-augment formulation, evidence-guided retrieval, and integrated budget control on 22-query frozen corpus. Offline validation complete; online acceptance and production cost measurement pending.
