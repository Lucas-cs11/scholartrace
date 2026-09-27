# Claim–Evidence Matrix

**Purpose**: Foundation for technical paper, PPT, and defense. Organized by claim → evidence → metric → artifact → limitation, not research chronology.

**Corpus**: pasa_realscholar_test_b3b570411ce2399c (22 queries, 184 Gold)

---

## Claim 1: CORE

**Claim**: Complex academic queries cannot rely on direct natural language for effective retrieval. Query understanding and search planning are core prerequisite capabilities for ScholarTrace.

### Evidence
- B0 baseline: raw query → OpenAlex → lexical ranking
- Result: F1 = 0.0095, 2/20 queries hit Gold
- Controlled ablation: Adding query planning (B1) increases recall to 1.0 on validated queries (q001, q002, q006)

### Metrics
- B0: final_unique_gold = 2/20 (historical corpus, 20 queries)
- B1 validated subset: recall = 1.0 (3/3 queries)

### Artifacts
- `eval/runs/B0_summary.json` (historical)
- Diagnostic: long natural language queries have low lexical overlap with Gold paper titles

### Limitation
- Query planning alone does not guarantee final F1 (requires ranking)
- B0/B1 use historical 20-query corpus, not directly comparable to current 184 Gold baseline

---

## Claim 2: SUPPORTING

**Claim**: Citation/reference/metadata expansion provides low cost-benefit in current evaluation task.

### Evidence
- FULL mode (with citation expansion): F1 = 0.0528, logical_api = 511
- NO_CITATION mode: F1 = 0.0618, logical_api = 87
- Citation/reference/metadata added Gold = 0

### Metrics
- Cost reduction: 511 → 87 logical API calls (-83%)
- F1 improvement: 0.0528 → 0.0618 (+17%)
- Incremental Gold from citation expansion: 0

### Artifacts
- Historical controlled experiments (M2 diagnostics)
- `eval/diagnostics/oracle_probe/` (citation reach analysis)

### Limitation
- **Context-specific**: "In current eval corpus and configuration"
- Citation graph may be valuable in different task contexts (citation recommendation, related work discovery)
- Do NOT generalize to "citation search has no value"

---

## Claim 3: SUPPORTING

**Claim**: Preserving original effective queries and augmenting with补充 queries is more robust than直接替换 existing search plans.

### Evidence
- M3-REPLACE: raw_unique_gold = 22, F1 = 0.0504
- M3-R (Preserve+Augment): raw_unique_gold = 25, F1 = 0.0664

### Metrics
- Gold improvement: 22 → 25 (+3 papers, +14%)
- F1 improvement: 0.0504 → 0.0664 (+32%)

### Artifacts
- `eval/runs/m3_sparse_plan_rescue/` (M3-REPLACE)
- `eval/runs/m3r_append/` (M3-R Preserve+Augment)

### Limitation
- Historical controlled ablation (corpus口径 different from current 184 Gold)
- Mark as: `historical_controlled_ablation`
- Mechanism validated, but absolute numbers not directly comparable to S2 baseline

---

## Claim 4: CORE / ROBUSTNESS

**Claim**: Comprehensive relevance ranking in current task cannot be fully replaced by retrieval frequency or lightweight deterministic semantic models, compared to generative semantic judgment.

### Evidence
Controlled reranker comparison (same candidate pool, M3-R frozen):

| Reranker | F1 | Type |
|----------|-----|------|
| LLM (deepseek-chat) | 0.0664 | Generative semantic |
| RRF (rank fusion) | 0.0241 | Frequency-based |
| MiniLM CE (matched-N) | 0.0598 | Deterministic semantic |
| BGE v2-m3 (matched-N) | 0.0588 | Deterministic semantic |

### Metrics
- LLM achieves highest F1 in frozen pool
- Deterministic models: 0.0588-0.0598 (11-12% lower than LLM)
- RRF: 0.0241 (64% lower than LLM)

### Artifacts
- `eval/runs/m4a1_rrf/`
- `eval/runs/m4a2_ce/`
- `eval/runs/m4a3_bge/`
- `eval/runs/m3r_append/` (LLM baseline)

### Limitation
- **LLM reranker has stochasticity** (temperature=0.0 reduces but doesn't eliminate)
- Deterministic rerankers provide **stability advantage**
- Some systematic misranking cases favor deterministic approaches
- Correct framing: "In current frozen candidate pool and config, LLM reranker achieves highest final F1, but deterministic methods offer stability and specific-case advantages."
- Do NOT claim: "LLM Reranker universally superior to all rerankers"

---

## Claim 5: CORE

**Claim**: Evidence-guided two-round search discovers papers not found in single-round retrieval.

### Evidence
- M5A iterative retrieval:
  - Round-1 raw_unique_gold: 25
  - Round-2 raw_unique_gold: 27
  - Incremental Gold: +2
  - Executed follow-ups: 62

### Metrics
- Gold improvement: 25 → 27 (+2, +8%)
- Cost: 62 follow-up queries executed
- Gold/API ratio: 2/62 ≈ 0.032 (low marginal efficiency)

### Artifacts
- `eval/runs/m5a_two_round/`
- `s11_budget_audit.json` (62/66 budget compliance)

### Limitation
- **Mechanism validated, but full-deployment cost-benefit insufficient**
- Round-2 adds significant API cost for limited Gold gain
- S2 design: FAST (default, no Round-2) + DEEP (optional, evidence-guided Round-2)
- F1 on 184 Gold corpus: FAST 0.0680 > DEEP 0.0641 (full Round-2 deployment didn't improve overall F1)

---

## Claim 6: CORE

**Claim**: ScholarTrace supports auditable autonomous iterative retrieval chains.

### Evidence
- Q29 (IMO math problems) trace:
  - Round-1 evidence → "Lean" (entity extraction)
  - Follow-up query: "Lean formal proof generation LLM"
  - Round-2 retrieval rank 8 → DeepSeek-Prover paper discovered

### Metrics
- Verified from frozen M5A SearchTrace
- Entity-driven follow-up successfully discovered out-of-Round-1 paper
- Rank position: 8 (entered rerank pool)

### Artifacts
- `eval/runs/m5a_two_round/m5a_round2_plans.jsonl` (frozen planner output)
- `eval/runs/m5a_two_round/m5a_round2_recall_cache.jsonl` (retrieval results)
- Q29 SearchTrace (Round-1 obs → Round-2 decision → discovery)

### Limitation
- Example from frozen trace, not hardcoded
- Demonstrates capability, not production-scale success rate
- Single case study, not statistical validation

---

## Claim 7: CORE

**Claim**: System integrates retrieval quality, cost, and reproducibility into unified engineering framework through budget control and SearchTrace.

### Evidence
System design features:
- Frozen query plans (offline reproducibility)
- Response cache (cost reduction)
- Prompt hash (planner version tracking)
- Config hash (experiment reproducibility)
- SearchTrace (complete execution log)
- No silent fallback (offline mode errors when cache missing)
- Round-2 budget enforcement (62/66 executed ≤ configured max 66)
- Gold leakage tests (static + runtime audit)

### Metrics
- Budget compliance: 62/66 ✅
- Offline replay: api_calls=0 (100% cache hit)
- Gold leakage tests: 15/15 PASS (S1) + 7/7 PASS (S1.1A)
- Reproducibility: frozen predictions hash-verified

### Artifacts
- `s1/pipeline.py` (budget enforcement)
- `s1/schemas.py` (SearchTrace schema)
- `s1/leakage.py` (Gold isolation)
- `tests/test_s1_integration.py` (integration tests)
- `tests/test_s11a_metrics.py` (metric/schema tests)
- `eval/runs/m5a_two_round/m5a_plan_meta.json` (prompt_hash)

### Limitation
- Offline acceptance PASS, online acceptance PENDING_LLM_BALANCE
- Production latency not measured (offline replay only)
- Logical API cost estimated from frozen plans, not measured end-to-end

---

## Claim Level Summary

| Claim | Level | Status |
|-------|-------|--------|
| 1. Query planning prerequisite | CORE | ✅ Validated (B0 vs B1) |
| 2. Citation expansion low ROI | SUPPORTING | ✅ Validated (controlled) |
| 3. Preserve+Augment robustness | SUPPORTING | ✅ Validated (ablation) |
| 4. LLM reranker quality vs deterministic | CORE/ROBUSTNESS | ✅ Validated (with limitations) |
| 5. Evidence-guided discovery | CORE | ✅ Mechanism validated, scale limited |
| 6. Auditable iterative chain | CORE | ✅ Case study (Q29) |
| 7. Integrated engineering framework | CORE | ✅ Validated (budget, trace, tests) |

---

## What NOT to Claim

❌ "LLM Reranker universally superior" → ✅ "LLM highest F1 in current frozen pool, with stochasticity tradeoff"

❌ "Citation search useless" → ✅ "Low ROI in current evaluation task configuration"

❌ "Evidence-guided Round-2 improves F1" → ✅ "Discovers new papers, but full-deployment cost-benefit insufficient; DEEP mode optional"

❌ "180 vs 184 Gold direct comparison" → ✅ "Mark historical experiments as different corpus, not directly comparable"

❌ "Production cost measured" → ✅ "Offline replay validated, production logical cost estimated/NOT_MEASURED"

---

## Technical Paper Structure Guidance

**NOT chronological research log** (M1 → M2 → M3 → M4 → M5):

**YES technical proof chain**:
1. Complex academic queries have term mismatch (Claim 1)
2. Query planning expands retrieval coverage (Claim 1)
3. Citation expansion costly, no Gold gain → budget reallocated to query planning (Claim 2)
4. Preserve+Augment more robust than Replace (Claim 3)
5. LLM reranker highest quality in eval, with deterministic alternatives for stability (Claim 4)
6. Evidence-guided Round-2 discovers out-of-Round-1 papers (Claim 5, 6)
7. Full Round-2 deployment low marginal gain → FAST/DEEP dual-mode design (Claim 5)
8. Unified framework: budget control + SearchTrace for quality/cost/reproducibility tradeoff (Claim 7)

**Avoid internal experiment codes in paper text**:
- ❌ "M3-R algorithm"
- ✅ "preserve-and-augment query formulation"

- ❌ "M5A iterative retrieval"
- ✅ "evidence-guided two-round retrieval"

**Cite artifacts for reproducibility**, use descriptive technical terms for claims.
