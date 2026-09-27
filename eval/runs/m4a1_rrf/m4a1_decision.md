# M4A1_DETERMINISTIC_RRF_BASELINE 决策报告

- 日期：2026-08-30 18:22；纯确定性 RRF，0 LLM、0 OpenAlex HTTP。
- 参考：M3-R observed best F1=0.0664（当前 production LLM Reranker 质量参考）。

## 1. 核心指标（Top-20）

| variant | mean F1 | mean P | mean R | raw | pool | final | final_inst | LLM | HTTP |
|---|---|---|---|---|---|---|---|---|---|
| M3-R_RRF | 0.0241 | 0.0159 | 0.0987 | 25 | 22 | 7 | 7 | 0 | 0 |
| M3.1_RRF | 0.0177 | 0.0114 | 0.0879 | 26 | 21 | 5 | 5 | 0 | 0 |

## 2. K 敏感度（diagnostic，不据此选 K）

| variant | K | F1 | P | R |
|---|---|---|---|---|
| M3-R_RRF | 5 | 0.007 | 0.0091 | 0.0057 |
| M3-R_RRF | 10 | 0.0322 | 0.0227 | 0.0903 |
| M3-R_RRF | 15 | 0.03 | 0.0212 | 0.0987 |
| M3-R_RRF | 20 | 0.0241 | 0.0159 | 0.0987 |
| M3.1_RRF | 5 | 0.0 | 0.0 | 0.0 |
| M3.1_RRF | 10 | 0.0223 | 0.0136 | 0.0795 |
| M3.1_RRF | 15 | 0.0223 | 0.0152 | 0.0879 |
| M3.1_RRF | 20 | 0.0177 | 0.0114 | 0.0879 |

## 3. Determinism

- M3-R_RRF: Top20 Jaccard=1.0 rank_identical=True f1_identical=True
- M3.1_RRF: Top20 Jaccard=1.0 rank_identical=True f1_identical=True

## 4. Gold recovery（M4-0 LOST_RERANKER，LLM 系统性 FN 能否被 RRF 恢复）

- LOST_RERANKER gold 总数=8；LLM(final 保留)=1；RRF top-20 保留=2；
- RRF 恢复率（LLM 漏但 RRF 进 top-20）=2/8 (25%)

## 5. Decision Gate（参考 F1=0.0664）

- best RRF F1 = 0.0241（M3-R_RRF），参考 LLM = 0.0664。
- **判定：RRF_ONLY_INSUFFICIENT**。不继续调 RRF 参数；下一步同样进入 M4A2_LIGHTWEIGHT_SEMANTIC_RERANKER。

## 6. 当前禁止（未违反）

- k=60 固定；query-type/Gold-aware 未加权；Planner/Retrieval/Prekeep/Safepass/LLM Prompt/temperature 均未改。
- Gold 仅在 evaluator 阶段读取；RRF 排序阶段 0 次读取 gold。未泄漏。

**本轮（M4A1）到此为止：STOP。不自动实现 M4A2。**