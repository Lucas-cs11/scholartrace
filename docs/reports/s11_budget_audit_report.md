# S1.1 Round-2 Budget Audit Report

**Verdict**: ✅ **ROUND2_BUDGET_ACCOUNTING = PASS**

---

## 执行摘要

**问题**: S1 report 显示 `round2_total=144`，但配置 `max_followup=3 × 22 queries = 66 theoretical max`。

**根因**: `round2_total` 是**指标命名错误**，实际含义是"final ranked output 中 retrieval_round=2 的论文数"，而非"执行的 follow-up query 数"。

**结论**: 
- 实际执行的 follow-up queries = **62** (≤ budget 66) ✅
- Budget enforcement 正确
- 需要修正指标命名：`round2_papers_total` → `final_round2_papers_total`

---

## 审计数据

### 配置
- `max_followup`: 3 per query
- `budget_max_new_searches`: 66
- Queries: 22 (frozen)

### 实际执行（offline replay with frozen M5A plans + cache）
- **Total planned follow-ups**: 66 (22 × 3)
- **Total executed follow-ups**: 62 (filtered 4 due to dup/generic)
- **Total filtered follow-ups**: 4
- **Budget compliance**: 62 ≤ 66 ✅

### Round-2 retrieval impact
- **Total newly discovered papers** (from Round-2 retrieval): 749
- **Total final Round-2 papers** (in ranked output, current replay): 0*

\* Current replay shows 0 because LLM reranker has insufficient balance (HTTP 402). The frozen eval JSON (generated when LLM was working) shows `round2_papers_total=144`, meaning 144 papers in final output had `retrieval_round=2`.

---

## Per-Query Breakdown (执行 follow-ups)

| Query ID | Planned | Executed | Filtered | Newly Discovered |
|----------|---------|----------|----------|------------------|
| RealScholarQuery_0 | 3 | 3 | 0 | 37 |
| RealScholarQuery_6 | 3 | 1 | 2 | 16 |
| RealScholarQuery_8 | 3 | 3 | 0 | 24 |
| RealScholarQuery_14 | 3 | 3 | 0 | 33 |
| RealScholarQuery_15 | 3 | 3 | 0 | 40 |
| RealScholarQuery_17 | 3 | 3 | 0 | 50 |
| RealScholarQuery_20 | 3 | 3 | 0 | 37 |
| RealScholarQuery_21 | 3 | 3 | 0 | 37 |
| RealScholarQuery_22 | 3 | 3 | 0 | 14 |
| RealScholarQuery_23 | 3 | 2 | 1 | 30 |
| RealScholarQuery_25 | 3 | 3 | 0 | 36 |
| RealScholarQuery_28 | 3 | 3 | 0 | 39 |
| RealScholarQuery_29 | 3 | 3 | 0 | 39 |
| RealScholarQuery_34 | 3 | 3 | 0 | 46 |
| RealScholarQuery_35 | 3 | 3 | 0 | 26 |
| RealScholarQuery_38 | 3 | 3 | 0 | 27 |
| RealScholarQuery_39 | 3 | 3 | 0 | 23 |
| RealScholarQuery_41 | 3 | 3 | 0 | 42 |
| RealScholarQuery_42 | 3 | 2 | 1 | 21 |
| RealScholarQuery_43 | 3 | 3 | 0 | 42 |
| RealScholarQuery_47 | 3 | 3 | 0 | 51 |
| RealScholarQuery_48 | 3 | 3 | 0 | 39 |

---

## 指标命名修正

### 旧字段（S1 原始，歧义）
- `round2_total` = 144

### 新字段（标准化，明确）
- `round2_executed_queries` = 62 （budget-relevant）
- `round2_newly_discovered_papers` = 749
- `final_round2_papers` = 144 （from frozen eval JSON）

---

## 技术细节

### Follow-up filter 逻辑（复用 M5A）
```python
from scripts.run_m5a import filter_followup

status = filter_followup(fu["query"], round1_queries)
# Returns: "keep" / "dup_round1_exact" / "dup_round1_similar" / 
#          "dup_round1_subset" / "generic" / "empty"
```

- **Executed**: `status == "keep"` or `status == "executed"`
- **Filtered**: Jaccard ≥ 0.8, subset, or generic (纯通用词)

### Budget enforcement
```python
# _round2() execution
executed_count = 0
for fu in followups:
    if fu.status not in {"keep", "executed"}:
        continue
    if executed_count >= budget_max_new_searches:
        break
    # ... retrieval ...
    executed_count += 1
```

---

## 验证

- [x] Budget compliance: 62 ≤ 66
- [x] Filter logic matches M5A
- [x] Metric naming issue identified
- [x] Per-query accounting traceable
- [x] Newly discovered papers counted
- [x] No silent budget violations

---

## 遗留问题

**LLM reranker insufficient balance**: Current replay cannot generate final ranked output. This does NOT invalidate:
- Budget accounting (retrieval-stage, independent of reranking)
- Frozen eval results (generated when LLM was working)
- S1 pipeline correctness

For full re-evaluation with reranking, LLM balance must be restored.

---

**结论**: ROUND2_BUDGET_ACCOUNTING = ✅ PASS

指标命名需修正，但 budget enforcement 正确。
