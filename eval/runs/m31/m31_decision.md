# M3.1_PLANNER_PROMPT_REVISION 决策报告

- 日期：2026-08-27 21:50（M3.1 全量 22/22-query 统一评测）
- M3_1_PLANNER_VERSION=2，prompt_hash=9fcfd4483e3b58fa…，plan_hash=75597f502c57a2ef…
- 对照：B0=`PASA_ASSOC_NO_CIT`（F1=0.0618 R=0.1339 raw=22）；M3-R_APPEND=`PASA_ASSOC_M3R_APPEND`（F1=0.0664 R=0.1492 raw=25）。
- 本轮唯一变量 vs M3-R：SparsePlanRescue Prompt（M3.1）。15 条 sparse = original_v2_subqueries + M3.1_rescue_subqueries（Preserve+Augment）；7 条 rich 原样。
- 冻结：assoc_safepass=True、Citation/Reference/Metadata OFF、OpenAlex 不变、top_k=20、Prekeep/Reranker 不变。

## 1. 三方保留漏斗（论文级 unique Gold papers，title_n；B0= safepass ON）

| 阶段 | B0 | M3-R_APPEND | M3.1 |
|---|---|---|---|
| raw unique Gold papers | 22 | 25 | **26** |
| pool unique Gold papers | 20 | 22 | 21 |
| final unique Gold papers | 13 | 16 | 14 |
| raw query-gold instances | 28 | 31 | 32 |
| raw→pool | 90.9% | 88.0% | 80.8% |
| pool→final | 65.0% | 72.7% | 66.7% |

## 2. 核心指标（三方）

| 指标 | B0 | M3-R_APPEND | M3.1 |
|---|---|---|---|
| mean F1 | 0.0618 | 0.0664 | 0.0519 |
| mean Precision | 0.0615 | 0.059 | 0.0522 |
| mean Recall | 0.1339 | 0.1492 | 0.1278 |
| OpenAlex logical calls | 87 | 0 | 48 |
| OpenAlex physical HTTP（新联网） | 87 | 0 | 48 |
| Reranker(LLM) calls | 77 | 95 | 95 |

## 3. Per-type CORE/ANCHOR/DISCOVERY（15 条 sparse；论文级 title_n）

| query_type | generated | executed | pruned | incremental Gold | Gold/executed |
|---|---|---|---|---|---|
| core | 15 | 15 | 0 | 1 | 0.067 |
| anchor | 29 | 28 | 1 | 2 | 0.071 |
| discovery | 15 | 15 | 0 | 1 | 0.067 |

## 4. Sparse 归因（论文级 title_n）

- preserved Gold（原始 sparse 子查询保留）= **4**
- incremental Rescue Gold（M3.1 rescue 子查询新增）= **4**
- Rescue Gold lost before pool = **1**；lost by reranker = **0**
- 新增 Rescue Gold raw→final 保留率 = 75%（3/4）

## 5. 决策 Gate（M3.1 raw_unique_gold，基线 B0=22 / M3-R=25；Step 12）

- **M3.1 raw_unique_gold_papers = 26**（B0 22 / M3-R 25）。
- **判定：QUERY_FORMULATION_TUNING_STOP**。raw 26~29：保留本版（相对 M3-R 有提升），但停止 Query Formulation Prompt tuning。

## 6. F1 判定（Step 13）

raw 提升（25→26）但 final F1 未同步提升（B0 0.0618 / M3-R 0.0664 / M3.1 0.0519）。
**NEXT = M4_RERANKER_RETENTION**（瓶颈在 final 保留，非 query formulation）。

## 7. 成本（Step 0/10/11；deterministic）

- **production_equivalent_logical_searches**：B0=87，M3-R=141，M3.1=145（orig 87 + rescue 58）
- **replay / cached_reused**：M3-R=52；M3.1=10（命中 pasa / M3 缓存，0 联网）
- **physical HTTP new**：M3-R=0；M3.1=48（<= 60）

## 8. Gold isolation 校验（Step 9）

- Production Planner 输入仅 question 文本；未接触 gold title/author/DOI/arXiv/abstract/Oracle probe/per-Gold taxonomy。
- 若发现泄漏 → EXPERIMENT_INVALID=true。本轮检查：Prompts 含 Gold isolation 规则；raw_counts 仅记录 LLM 输出结构，无 gold 信息。
- Gold 仅在 evaluator（TraceRecorder）于 retrieval 后读取。

**本轮（M3.1）到此为止：无论结果如何，停止 Query Formulation Prompt tuning；完成后 STOP，不自动进入 M4。**