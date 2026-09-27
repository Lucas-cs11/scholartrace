# M3-R_APPEND 决策报告（Preserve+Augment，纯离线重放）

- 日期：2026-08-27 21:13（M3-R 全量 22/22-query 离线重放）
- 对照 B0：`PASA_ASSOC_NO_CIT`（F1=0.0618）；M3-REPLACE：`PASA_ASSOC_M3_SPARSE_RESCUE`（raw=22）。
- **唯一变量：replace -> append**。15 条 sparse = original_v2_subqueries + cached_M3_rescue_subqueries；7 条 rich 原样。不重新生成 Rescue Plan / 不修改 M3 Planner Prompt。
- 冻结：assoc_safepass=True、Citation/Reference/Metadata OFF、OpenAlex 不变、top_k=20、Prekeep/Reranker 不变。
- 离线重放校验：OpenAlex api_calls=0 / physical HTTP=0（全部响应来自缓存；任一 miss 即 STOP）。

## 1. 三方保留漏斗（论文级 unique Gold papers，title_n；B0= safepass ON）

| 阶段 | B0 | M3-REPLACE | M3-APPEND |
|---|---|---|---|
| raw unique Gold papers | 22 | 22 | **25** |
| pool unique Gold papers | 20 | 20 | 22 |
| final unique Gold papers | 13 | 13 | 16 |
| raw query-gold instances | 28 | 28 | 31 |
| raw→pool | 90.9% | 90.9% | 88.0% |
| pool→final | 65.0% | 65.0% | 72.7% |

## 2. 核心指标（三方）

| 指标 | B0 | M3-REPLACE | M3-APPEND |
|---|---|---|---|
| mean F1 | 0.0618 | 0.0504 | 0.0664 |
| mean Precision | 0.0615 | 0.0487 | 0.0590 |
| mean Recall | 0.1339 | 0.1253 | 0.1492 |
| OpenAlex logical calls | 87 | 52 | 0 |
| OpenAlex physical HTTP（本轮新联网） | 87 | 52 | 0 |
| Reranker(LLM) calls | 77 | 95 | 95 |

- M3-R 本轮新联网 OpenAlex physical HTTP = **0**（cache-first 全命中；复用了 M3 已生成的 52 条 cached HTTP response，0 自动联网补跑）。

## 3. Sparse 归因（15 条；论文级 title_n）

- preserved Gold（原始 sparse 子查询保留，append 后仍在 raw）= **4**
- incremental Rescue Gold（rescue 子查询新增、原始未命中）= **3**
- Rescue Gold lost before pool（raw 命中但 prekeep 丢弃）= **0**
- Rescue Gold lost by reranker（pool 但 final 丢弃）= **0**

## 4. 决策 Gate（append raw_unique_gold，基线 22）

- **append raw_unique_gold_papers = 25**，incremental Rescue Gold = 3。
- **判定：QUERY_FORMULATION_PARTIAL_SIGNAL**。下一阶段允许执行：一次且仅一次 M3.1 Planner Prompt Revision。

- 新增 Rescue Gold raw→final 保留率 = 100%（3/3）。
  → 新增 Rescue Gold 的 raw→final 保留较高：当前主要瓶颈仍是 **retrieval / query formulation**，而非 reranker。不自动进入 M4。

## 5. 范围遵守声明

- src/search.py / src/planner.py / src/planner_rescue.py 均未改动。本轮禁止：ANCHOR/DISCOVERY prompt revision、新 query type、Quality Selector、Adaptive Retrieval、Reranker modification。
- Cache first：原始 v2/rich 读 _pasa_recall_cache，M3 Rescue 读 M3 response cache；OpenAlex physical HTTP=0，无自动联网补跑。
- 统一口径：raw/pool/final 同时输出 instances(canonical_id) 与 unique Gold papers(title_n)。

**本轮（M3-R）到此为止：完成后 STOP，等待用户批准后才进入 M3.1 / M4，不自行执行。**