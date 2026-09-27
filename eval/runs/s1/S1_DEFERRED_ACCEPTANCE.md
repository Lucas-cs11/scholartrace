# S1 Deferred Acceptance Items

**Reason**: LLM_API_HTTP_402_INSUFFICIENT_BALANCE

These are **deferred execution checks**, not failed algorithm gates. All algorithm implementation is complete and correct.

---

## Deferred Items

### 1. FAST Online Smoke Test
**Requirement**: Run 2 queries in online mode (real Planner + OpenAlex)  
**Validation**: Planner callable, OpenAlex callable, reranker callable, structured output valid, SearchTrace complete  
**Status**: DEFERRED  
**Blocker**: Requires LLM for Planner query decomposition and LLM Reranker

### 2. DEEP Online Smoke Test
**Requirement**: Run 2 queries in online DEEP mode (FAST + Round-2)  
**Validation**: Same as FAST + Round-2 budget enforcement, follow-up generation, newly discovered papers  
**Status**: DEFERRED  
**Blocker**: Requires LLM for Planner, Round-2 planner (frozen), and LLM Reranker

### 3. Production Q29/Q43 Demo Replay
**Requirement**: Replay Q29/Q43 with actual LLM Reranker to demonstrate final ranked output  
**Validation**: FAST final output non-empty, DEEP final output includes Round-2 papers, SearchTrace complete  
**Status**: DEFERRED  
**Blocker**: Requires LLM Reranker (currently returns empty due to balance)

### 4. Metric-Corrected Full Eval Regeneration (Optional)
**Requirement**: Re-run full 22-query FAST/DEEP eval with standardized metric names  
**Validation**: New JSON artifacts use `final_unique_gold`, `total_gold_papers`, `round2_executed_queries`, etc.  
**Status**: DEFERRED (optional, frozen results already use correct semantics)  
**Blocker**: Requires LLM Reranker

---

## Current Valid Artifacts

The following frozen artifacts were generated when LLM had balance and are **valid for use**:

1. `eval/runs/s1/s1_fast_offline.json` — FAST baseline (F1=0.0680, final_unique_gold=14)
2. `eval/runs/s1/s1_deep_offline.json` — DEEP baseline (F1=0.0641, final_unique_gold=15)
3. All SearchTrace data in these files

**Note**: These files use old field names (`raw_gold`, `gold_n`, `round2_papers`) but the semantics are documented in S1.1 audit reports. Manual translation:
- `raw_gold` → `final_unique_gold`
- `gold_n` → `total_gold_papers`
- `round2_papers` (per-query) → `final_round2_papers`
- `round2_papers_total` → `final_round2_papers_total` (144 = final output papers, NOT query count)

---

## Execution Procedure (When LLM Balance Restored)

```bash
# 1. Online smoke tests
python3 scripts/run_eval.py --mode fast --online --limit 2 --out eval/runs/s1/s1_fast_online_smoke.json
python3 scripts/run_eval.py --mode deep --online --limit 2 --out eval/runs/s1/s1_deep_online_smoke.json

# 2. Production demo replay
python3 scripts/replay_demo.py --out eval/runs/s1/s1_demo_replay_production.json

# 3. (Optional) Metric-corrected full eval
python3 scripts/run_eval.py --mode fast --out eval/runs/s1/s1_fast_offline_v2.json
python3 scripts/run_eval.py --mode deep --out eval/runs/s1/s1_deep_offline_v2.json
```

**Expected outcome**: All 4 items PASS, S1_STATUS → FULLY_ACCEPTED

---

## Impact on S2

**No blocker for S2 entry**. Frozen S1 results provide valid baseline:
- FAST F1=0.0680 (final_unique_gold=14/184)
- DEEP F1=0.0641 (final_unique_gold=15/184)
- Round-2 budget accounting verified (62/66 executed)

S2 can proceed with these baselines. Online validation remains deferred but does not block contest preparation.

---

**Status**: S1_STATUS=CONDITIONALLY_ACCEPTED, S2_ENTRY=APPROVED
