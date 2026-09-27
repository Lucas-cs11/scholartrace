# M2_SAFEPASS_ABLATION 决策报告（保送开关消融，offline replay）

- 日期：2026-08-27 20:26（M2 全量 22-query，offline replay）
- M2 实验：`PASA_ASSOC_M2_DENSITY`，22/22 条完成。**回放 v2 冻结 plan（89 条 subquery），assoc_safepass=False，0 次 OpenAlex 联网**（recall 全命中 _pasa_recall_cache.jsonl）。
- Retriever/Prekeep/Reranker/top-k=20 冻结；引文扩展关闭。

## 1. 保留漏斗（论文级 unique Gold，title_n 去重；M3 gate 口径）

| 阶段 | safepass ON（v2 实测） | safepass OFF（M2 回放） | Δ |
|---|---|---|---|
| raw unique Gold papers | 22 | 22 | +0 |
| pool unique Gold papers | 20 | 10 | -10 |
| final unique Gold papers | 13 | 8 | -5 |
| raw→pool 保留率 | 90.9% | 45.5% | -45.5% |
| pool→final 保留率 | 65.0% | 80.0% | +15.0% |

**实例级（canonical_id/DOI，含同篇多 DOI 形态）**：
raw_query_gold_instances = 28，pool_query_gold_instances = 13，final_query_gold_instances = 11
（论文级 raw 22 篇 ↔ 实例级 28 条，Phase 2.5 口径一致：28 实例 ↔ 22 论文）

## 2. 核心指标

| 指标 | B0 | M2 (safepass OFF) | Δ |
|---|---|---|---|
| F1 | 0.0618 | 0.0485 | -0.0132 |
| Precision | 0.0615 | 0.0544 | -0.0071 |
| Recall | 0.1339 | 0.0663 | -0.0676 |

## 3. 轨贡献（first-seen 归因；assoc=联想词密度轨，regular=常规轨）

| track | executed_queries | new_unique_gold | incremental_gold/query |
|---|---|---|---|
| assoc | 42 | 20 | 0.476 |
| regular | 47 | 8 | 0.170 |

## 4. 保送价值判定

- **保送对 raw 无影响**（raw 在 prekeep 之前计数，_build_rerank_pool 只影响 pool）：M2 raw = OFF = ON = 22，完全一致。
- **保送对 raw→pool 保留**（保送直接杠杆）：OFF=45.5% vs ON=90.9%，Δ=-45.5%pp。
- **pool→final 保留**（受 pool 规模混杂，仅参考）：OFF=80.0% vs ON=65.0%。
- **净 final Gold**：ON=13 vs OFF=8（保送净增 +5）；F1 ON=0.0618 vs OFF=0.0485。

**判定：保送明显提高 raw→pool Gold 保留（Δ≥10pp）→ MVP 保留 assoc_safepass=True。**

## 5. 范围遵守声明

- src/search.py 改动：仅新增 `assoc_safepass` 开关（默认 True 保持原行为；M2 置 False）。冻结参数/prekeep/reranker/top-k 未动。
- gold isolation：不回放 gold；planner 不参与（回放冻结 v2 plan）。
- **offline replay 校验：OpenAlex api_calls == 0（零联网）**；仅 LLM 精排调用。

**本轮到此为止：等待用户批准后才进入下一阶段（M3_SPARSE_PLAN_RESCUE）。**