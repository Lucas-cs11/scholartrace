# S1.1 Gold Corpus Audit Report

**Verdict**: ✅ **EVAL_CORPUS_VERSION = FROZEN**

---

## 执行摘要

**问题**: Historical reports mention "180 Gold", but S1 eval shows "184 Gold".

**结论**: 
- **180**: Unknown historical subset (possibly earlier 22-query selection or different test split)
- **184**: S1 frozen 22 queries from PaSa RealScholarQuery test.jsonl (confirmed)
- **791**: Full 50 queries in test.jsonl

**决策**: 
- S1 及后续实验统一使用 **184 Gold (frozen 22 queries)**
- Corpus version: `pasa_realscholar_test_b3b570411ce2399c`

---

## 数据源

### Data File
- **Path**: `data/benchmarks/pasa/RealScholarQuery/test.jsonl`
- **Hash (SHA256[:16])**: `b3b570411ce2399c`
- **Total queries**: 50
- **Frozen 22 queries**: RealScholarQuery_{0,6,8,14,15,17,20,21,22,23,25,28,29,34,35,38,39,41,42,43,47,48}

### Gold Count (evaluator logic: `len(match_gold(q))`)
- **Full 50 queries**: 791 Gold papers
- **Frozen 22 queries**: **184 Gold papers** ✅
- **Corpus fingerprint (SHA256[:16] of sorted identity keys)**: `2184b6a5082ccc0f`

---

## 180 vs 184 分析

### Hypothesis 1: Different query subset
Historical 180 may refer to:
- An earlier 22-query selection with slightly different queries
- A 20-query subset (challenges_v1.jsonl shows 20 Gold instances, but that's a different eval format)
- Different de-duplication logic

### Hypothesis 2: Data version
PaSa benchmark may have been updated. Without access to the historical eval's exact data file, we cannot definitively reconcile the discrepancy.

### Decision
- **Do NOT retroactively change historical reports** (180 remains 180 in context)
- **S1 forward uses 184** as the authoritative Gold count for the frozen 22 queries
- **Technical paper must note**: "Frozen 22-query subset contains 184 unique Gold papers (full 50-query set: 791)"

---

## Evaluator Logic

### `match_gold(query)` from `eval.harness`
For each Gold paper in `query["gold"]`, generates identity keys:
1. `openalex:<openalex_id>` (if present)
2. `doi:<normalized_doi>` (if present, lowercase, strip `https://doi.org/`)
3. `title:<normalized_title>` (lowercase, whitespace-collapsed)
4. `title_n:<letters_only>` (alphanumeric only, lowercase)

Returns a set of these identity key sets. `len(match_gold(q))` = paper-level unique Gold count.

### Query-Gold Instances vs Paper-Level Gold
- **Query-gold instances** (sum of `len(q["gold"])` per query): 791 (frozen 22 queries includes this field inline)
- **Evaluator paper-level Gold** (`sum(len(match_gold(q)))`): 184 (after identity de-duplication)

The evaluator uses **paper-level** for P/R/F1 calculation.

---

## Corpus Fingerprint

For reproducibility:
```
EVAL_CORPUS_VERSION = pasa_realscholar_test_b3b570411ce2399c
FROZEN_22_GOLD_PAPERS = 184
FROZEN_22_QUERIES = [
  RealScholarQuery_0, RealScholarQuery_6, RealScholarQuery_8,
  RealScholarQuery_14, RealScholarQuery_15, RealScholarQuery_17,
  RealScholarQuery_20, RealScholarQuery_21, RealScholarQuery_22,
  RealScholarQuery_23, RealScholarQuery_25, RealScholarQuery_28,
  RealScholarQuery_29, RealScholarQuery_34, RealScholarQuery_35,
  RealScholarQuery_38, RealScholarQuery_39, RealScholarQuery_41,
  RealScholarQuery_42, RealScholarQuery_43, RealScholarQuery_47,
  RealScholarQuery_48
]
```

---

## 验证

- [x] Data source hash recorded
- [x] Frozen 22 Gold count confirmed: 184
- [x] Evaluator logic traced
- [x] 180 vs 184 discrepancy analyzed
- [x] Corpus fingerprint generated
- [x] Decision: do not retroactively unify

---

## 技术论文引用格式

```
我们在 PaSa RealScholarQuery benchmark 的 22-query frozen subset 上评测，
该子集包含 184 篇 unique Gold 论文（full 50-query set: 791）。
Historical baselines 报告的 180 Gold 可能基于不同的 query selection 或数据版本。

Corpus version: pasa_realscholar_test_b3b570411ce2399c
```

---

**结论**: EVAL_CORPUS_VERSION = ✅ FROZEN

S1 forward 统一使用 184 Gold (frozen 22 queries)。
