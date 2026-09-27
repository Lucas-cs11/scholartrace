# M3_SPARSE_PLAN_RESCUE 决策报告

- 日期：2026-08-27 20:49（M3 全量 22/22-query 统一评测）
- M3_PLANNER_VERSION=1，plan_hash=945ccdffe1bf1f73…
- 对照 B0：`PASA_ASSOC_NO_CIT`（F1=0.0618 P=0.0615 R=0.1339 api=87 llm=77）——未重跑，读基线报告
- M3 实验：`PASA_ASSOC_M3_SPARSE_RESCUE`。**7 条 rich plan 冻结原样**（含 v2 assoc 锚点），**15 条 sparse plan 用 SparsePlanRescue 重写**（≤4：core×1+anchor×1-2+discovery×0-1）。
- 冻结：assoc_safepass=True（KEEP_SAFEPASS）、Citation/Reference/Metadata OFF、OpenAlex 不变、top_k=20、Prekeep 不变、Reranker 不变。
- 缓存/网络：rich/既有子查询全读 _pasa_recall_cache（0 联网）；仅新 rescue 子查询请求 OpenAlex。

## 1. 检索原始召回（论文级 unique Gold，title_n 去重；M3 gate 口径）

- **raw_unique_gold_papers = 22**（B0 基线 22；15 条 sparse 原贡献 ~0）
- **raw_query_gold_instances = 28**（canonical_id/DOI 实例级）
- pool_unique_gold = 20（survived_prekeep，论文级）
- final_unique_gold = 13（final top-20，论文级）
- raw→pool = 90.9%，pool→final = 65.0%

## 2. Rescue 增益（15 条 sparse query）

- rescue 新增 unique Gold 论文（未命中原始 sparse plan 的）：**3**
- 其中 RETRIEVAL_GAIN（新 Gold 最终进 top-k）= **3**；RETRIEVAL_GAIN_BUT_RERANK_LOSS（raw 命中原 Gold 但被 prekeep/reranker 丢弃）= **0**

| query_id | rescue前 | rescue后 | new | CORE | ANCHOR | DISCOVERY | calls | 增量/call |
|---|---|---|---|---|---|---|---|---|
| RealScholarQuery_8 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| RealScholarQuery_14 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| RealScholarQuery_17 | 0 | 1 | 1 | 0 | 0 | 1 | 3 | 0.333 |
| RealScholarQuery_20 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| RealScholarQuery_21 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| RealScholarQuery_22 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| RealScholarQuery_28 | 0 | 1 | 1 | 0 | 1 | 0 | 4 | 0.25 |
| RealScholarQuery_29 | 3 | 1 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| RealScholarQuery_34 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| RealScholarQuery_38 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| RealScholarQuery_39 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.0 |
| RealScholarQuery_41 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| RealScholarQuery_42 | 0 | 1 | 1 | 0 | 1 | 0 | 4 | 0.25 |
| RealScholarQuery_43 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0.0 |
| RealScholarQuery_48 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0.0 |

## 3. 核心指标（vs B0）

| 指标 | B0 | M3 | Δ |
|---|---|---|---|
| F1 | 0.0618 | 0.0504 | -0.0113 |
| Precision | 0.0615 | 0.0487 | -0.0128 |
| Recall | 0.1339 | 0.1253 | -0.0086 |
| logical API calls/query | 4.0 | 2.4 | |
| physical HTTP attempts（全量） | 87 | 52 | |
| LLM calls/query | 3.5 | 4.3 | |
| 总延迟（ms） | - | 479165 | |

## 4. 决策 Gate（论文级 raw，基线 22/180）

- ≥40 → STRONG_RETRIEVAL_SUCCESS（立即冻结 M3 Planner）；≥35 → MVP_RETRIEVAL_SUCCESS；≥30 → QUERY_FORMULATION_SIGNAL；<30 → QUERY_FORMULATION_INSUFFICIENT（只允许 1 次 prompt 修订）。
- **判定：QUERY_FORMULATION_INSUFFICIENT**（raw_unique_gold_papers = 22）

raw 未达标：只允许 1 次 M3 Planner prompt 修订，若仍不达标回 B0 基线。

## 5. 范围遵守声明

- src/search.py 未改动；src/planner.py 未改动。新增 src/planner_rescue.py + scripts/run_m3_sparse.py。
- Gold isolation：rescue planner 输入仅 question 文本；严禁基于 Gold 生成论文标题/作者/DOI/arXiv ID。
- 只解决 15 条 sparse plan 的 Query Formulation；7 条 rich plan 冻结原样，未重新调用。
- 联网仅新 rescue 子查询（rescue_new_total 对应子查询），既有 v2 子查询 0 联网重跑。

**本轮（M3）到此为止：完成 22 条统一评测后 STOP，等待用户批准后才进入 M4，不自行进入。**