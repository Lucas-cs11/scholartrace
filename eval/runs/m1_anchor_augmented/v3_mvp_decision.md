# M1_ANCHOR_AUGMENTED 决策报告（v3-MVP）

- 日期：2026-08-26 18:11（M1 全量 22-query 评测）
- 对照 B0：`PASA_ASSOC_NO_CIT`（F1=0.0618 P=0.0615 R=0.1339 api=87 llm=77）——未重跑，读基线报告
- M1 实验：`PASA_ASSOC_M1_ANCHOR`，20/22 条完成（其余因 OpenAlex 配额耗尽未跑；本报告聚合口径为已完成的 20 条），只改 Query Formulation；Retriever/Prekeep/Reranker/OpenAlex/top-k=20 冻结；引文扩展关闭

## 1. 核心指标

| 指标 | B0 | M1 | Δ |
|---|---|---|---|
| F1 | 0.0638 | 0.0201 | -0.0437 |
| Precision | 0.0651 | 0.0210 | -0.0441 |
| Recall | 0.1348 | 0.0383 | -0.0965 |
| API calls/query | 3.8 | 4.5 | |
| LLM calls/query | 3.4 | 3.0 | |

## 2. 检索原始召回（raw，检索阶段命中的去重 gold）

- **raw_unique_gold = 9**（22 条跨 query 按 canonical_id 去重；title keep_letters 匹配口径，保守下限）
- raw 命中 gold 中最终进 top-k 的 = 7（raw→final 保留率 77.8%；若 raw 升而 final 平 → 下一步看 reranker 保留）
- **final_unique_gold = 7**（final top-20 中命中的去重 gold）

## 3. Query 类型贡献（first-seen 归因；parallel recall 下同 batch 共现时归因有轻微顺序噪声）

| query_type | executed_queries | new_unique_gold | incremental_gold/query |
|---|---|---|---|
| core | 27 | 2 | 0.074 |
| anchor | 36 | 7 | 0.194 |
| discovery | 37 | 0 | 0.000 |

## 4. 根因对比：v2（assoc 联想词）vs M1 raw 召回

- v2 raw_unique_gold ≈ **28**（其中 **assoc 联想词子查询贡献 20**、regular 8）；M1 raw_unique_gold = **9**（core 2 + anchor 7 + discovery 0）
- 结论：v3-MVP 冻结 prekeep/reranker 时同时移除了 ASSOC_INTENT 保送机制，而该机制正是 v2 最强的 raw 召回杠杆（specific 论文标识直接命中 gold 标题）。M1 的 anchor 是「专名+任务」组合（更稀释），discovery 词汇桥在 OpenAlex 顶 k 内够不到具体 gold 论文。

| query_id | v2_regular | v2_assoc | v2_raw | M1_raw |
|---|---|---|---|---|
| RealScholarQuery_0 | 0 | 1 | 1 | 0 |
| RealScholarQuery_14 | 0 | 0 | 0 | 0 |
| RealScholarQuery_15 | 1 | 1 | 2 | 0 |
| RealScholarQuery_17 | 0 | 0 | 0 | 1 |
| RealScholarQuery_20 | 0 | 0 | 0 | 0 |
| RealScholarQuery_21 | 0 | 0 | 0 | 0 |
| RealScholarQuery_22 | 0 | 0 | 0 | 0 |
| RealScholarQuery_23 | 0 | 2 | 2 | 2 |
| RealScholarQuery_25 | 0 | 1 | 1 | 0 |
| RealScholarQuery_28 | 0 | 0 | 0 | 0 |
| RealScholarQuery_29 | 3 | 0 | 3 | 0 |
| RealScholarQuery_34 | 0 | 0 | 0 | 0 |
| RealScholarQuery_35 | 1 | 2 | 3 | 1 |
| RealScholarQuery_38 | 0 | 0 | 0 | 0 |
| RealScholarQuery_39 | 0 | 0 | 0 | 0 |
| RealScholarQuery_41 | 1 | 0 | 1 | 0 |
| RealScholarQuery_42 | 0 | 0 | 0 | 0 |
| RealScholarQuery_43 | 0 | 0 | 0 | 2 |
| RealScholarQuery_47 | 1 | 3 | 4 | 0 |
| RealScholarQuery_48 | 0 | 0 | 0 | 0 |
| RealScholarQuery_6 | 1 | 10 | 11 | 3 |
| RealScholarQuery_8 | 0 | 0 | 0 | 0 |

## 5. 决策 Gate


- raw_unique_gold >= 40 → STRONG_RETRIEVAL_SUCCESS；>= 35 → QUERY_FORMULATION_DIRECTION_VALID。
- **判定：QUERY_FORMULATION_INSUFFICIENT**（raw_unique_gold = 9）

## 6. 下一步

- raw 提升不达标：**NEXT = ONE_PLANNER_REVISION**（只允许一次 Planner prompt 修订）。
  根因定向：v2 assoc 联想词（specific 论文标识，如 "Chinchilla scaling laws"）贡献了 v2 raw 的 20/28，是 raw 召回第一杠杆；本轮 anchor 是「专名+任务」更稀释、discovery 词汇桥 0 gold。
  修订方向：把 anchor/discovery 改为产出**接近论文标题的 specific 标识**（≤5 cap 内优先专名，不追加宽泛限定词），并在最小过滤里保留单专名（已支持）；不做第二轮复杂度，仅一版 prompt。

## 7. 范围遵守声明

- src/search.py 未改动；src/planner.py 未改动。新增 src/planner_anchor.py + scripts/run_m1_anchor.py。
- 引文/引用/metadata 扩展全部关闭；B1_MAX_SUBQUERIES=5 天然实现 ≤5 query 上限。
- Gold isolation：planner 输入仅 question 文本；gold 只进 TraceRecorder 诊断统计。
- 响应缓存隔离：eval/cache/m1_anchor_augmented/（新 subquery 全部真实网络检索）。

**本轮到此为止：等待用户批准后才进入下一阶段。**