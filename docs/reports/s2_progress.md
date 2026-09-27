# S2 Contest Finalization

**Status**: 🔄 IN PROGRESS  
**Date**: 2026-08-31

---

## Global Freeze Status

```text
ALGORITHM_RESEARCH_FREEZE = true
RERANKER_RESEARCH_FREEZE = true
QUERY_FORMULATION_RESEARCH_FREEZE = true
ITERATIVE_RETRIEVAL_RESEARCH_FREEZE = true
```

**No algorithm modifications allowed in S2.**

---

## S1 Entry Status

```text
S1_IMPLEMENTATION_COMPLETE = true
S1_OFFLINE_ACCEPTANCE = PASS
S1_ONLINE_ACCEPTANCE = PENDING_LLM_BALANCE
S1_STATUS = CONDITIONALLY_ACCEPTED
S2_ENTRY = APPROVED
```

---

## Official Evaluation Corpus

```text
EVAL_CORPUS_VERSION = pasa_realscholar_test_b3b570411ce2399c
DATA_HASH = b3b570411ce2399c
CORPUS_FINGERPRINT = 2184b6a5082ccc0f
FROZEN_QUERIES = 22
TOTAL_GOLD_PAPERS = 184
```

All S2 reports, README, technical papers, PPT, and defense presentations must use this official corpus.

Historical "180 Gold" experiments are preserved but marked as `historical experimental corpus`, not directly comparable to current 184 Gold baseline.

---

## Official Metric Definitions

### Deprecated (forbidden in S2)
- ~~`raw_gold`~~
- ~~`gold_n`~~
- ~~`round2_total`~~
- ~~`retrieved`~~
- ~~`final`~~

### Standardized Lifecycle
- `retrieval_raw_unique_gold`
- `retrieval_raw_gold_instances`
- `rerank_pool_unique_gold`
- `final_unique_gold`
- `final_gold_instances`
- `total_gold_papers`

### Round-2 (DEEP)
- `round2_planned_queries`
- `round2_filtered_queries`
- `round2_executed_queries` (invariant: ≤ 66)
- `round2_newly_discovered_papers`
- `final_round2_papers`

---

## Official Contest Modes

### FAST (Default)
```
Question → Planner → Regular+Assoc Retrieval → OpenAlex
→ Identity Normalization → Candidate Pool → LLM Reranker
→ Structured Results → SearchTrace
```

**Frozen offline reference**:
- F1 = 0.0680
- final_unique_gold = 14 / 184

### DEEP (Complex Query Mode)
```
FAST Round-1 → SearchObservation → Evidence Analysis
→ Frozen M5A Round-2 Planner → Follow-up Retrieval
→ Candidate Merge → LLM Reranker → Structured Results → SearchTrace
```

**Frozen offline reference**:
- F1 = 0.0641
- final_unique_gold = 15 / 184
- round2_executed_queries = 62 / 66
- round2_newly_discovered_papers = 749
- final_round2_papers = 144

**Note**: DEEP F1 < FAST is preserved as-is. Technical paper must state: "Evidence-guided two-round retrieval discovers papers not found in Round-1, but full-deployment does not improve overall F1 on current test set. FAST is the default mode; DEEP provides optional deep search capability for complex queries."

---

## S2 Tasks

### S2-A: Final Frozen Evaluation ✅
- Use existing frozen artifacts (no new API/LLM consumption)
- Recompute with standardized metrics
- Generate: `s2_final_metrics_fast.json`, `s2_final_metrics_deep.json`, `s2_per_query_metrics.csv`

### S2-B: Efficiency Evaluation
- FAST: LLM calls/q, API calls/q, tokens/q, latency metrics
- DEEP: Same + Round-2 incremental costs
- Distinguish: offline replay (physical=0) vs production-equivalent logical cost

### S2-C: Structured Output Validation
- Schema validation on all 22 queries × 2 modes
- Verify factual metadata from academic sources (not LLM-generated)
- Target: schema_valid_rate = 100%

### S2-D: Claim-Evidence Matrix
- Organize verified claims with artifact support
- No research log, only technical proof structure

### S2-E: Failure & Limitation Matrix
- Document known limitations honestly

### S2-F: Submission Audit
- No .env, API keys, Gold in production runtime, absolute paths
- Tests PASS

### S2-G: Paper Fact Sheet
- Only artifact-supported facts

### S2-H: Defense Evidence Pack
- Claims + evidence + suggested visualizations

---

## S2 Gate (In Progress)

| Gate | Status |
|------|--------|
| CORPUS_FROZEN | ✅ PASS |
| METRIC_SEMANTICS | ✅ PASS |
| PREDICTION_REPRODUCIBILITY | 🔄 TODO |
| FAST_FINAL_EVAL | 🔄 TODO |
| DEEP_FINAL_EVAL | 🔄 TODO |
| EFFICIENCY_REPORT | 🔄 TODO |
| STRUCTURED_OUTPUT | 🔄 TODO |
| SEARCHTRACE | 🔄 TODO |
| IDENTITY_AUDIT | 🔄 TODO |
| GOLD_LEAKAGE | ✅ PASS (from S1) |
| SUBMISSION_AUDIT | 🔄 TODO |
| OFFLINE_REPRODUCTION | 🔄 TODO |
| TESTS | ✅ PASS (22/22 from S1.1A) |
| CLAIM_EVIDENCE_MATRIX | 🔄 TODO |
| PAPER_FACT_SHEET | 🔄 TODO |
| DEFENSE_EVIDENCE_PACK | 🔄 TODO |

---

**Progress**: Starting S2-A (Final Frozen Evaluation)
