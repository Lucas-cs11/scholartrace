# M4A2_1_OUTPUT_SELECTION_AUDIT 决策报告

- 日期：2026-08-30 19:39；全程离线：OpenAlex HTTP=0、LLM calls=0、CE inference=0。
- 数据源：m4a2_ce_rankings.csv（冻结 CE 排序/分数）+ M3-R trace final_reranked（LLM final 输出）+ plan/recall cache（离线重建 pool）。
- 交叉验证：LLM F1 复算=0.0664（参考 0.0664）；CE Top20 F1 复算=0.0503（参考 0.0503）。


## 1. LLM output cardinality（Section 2）

- 22 queries LLM final result count：min=3 p25=9 median=10.0 mean=12.18 p75=18 max=20。
- LLM 是 **Ranking + Variable-length Selection**（score>=0.35, min_keep=3, max_results=20），非固定 20 篇。

## 2. Matched-cardinality（Section 3，同 prediction budget）

- LLM variable-N：mean F1=0.0664，final unique Gold=16，instances=16。
- CE matched-N（N=LLM 每 query 输出数）：mean F1=0.0598，final unique Gold=14，instances=14。

## 3. Ranking diagnostics（Section 4，binary relevance，@20 因冻结 Top-20 输出）

| metric | CE | LLM |
|---|---|---|
| mrr@20 | 0.1838 | 0.3693 |
| map@20 | 0.0505 | 0.1198 |
| recall@5 | 0.0953 | 0.1523 |
| recall@10 | 0.1597 | 0.1746 |
| recall@20 | 0.1673 | 0.1803 |
| ndcg@5 | 0.099 | 0.2068 |
| ndcg@10 | 0.1144 | 0.1906 |
| ndcg@20 | 0.1129 | 0.1857 |

## 4. Score-gap（Section 5，唯一允许的 adaptive cutoff）

- M4A2_1_SCORE_GAP：i∈[3,min(19,n-1)] 取最大 gap，tie 取更小 i；min=3 max=20。
- mean F1=0.0425，final unique Gold=8，instances=8。
- cardinality：min=3 max=17（详见 m4a21_score_gap_predictions.csv）。

## 5. Determinism（Section 7）

- selected_ids_identical=True；cardinality_identical=True；f1_identical=True；queries=22。
- → DETERMINISM PASSED。

## 6. 对照表（Section 8）

| Variant | Selection | F1 | P | R | final Gold | Gen LLM calls |
|---|---|---|---|---|---|---|
| M3-R LLM | LLM threshold | 0.0664 | — | — | 16 | 77 |
| RRF | Top20 | 0.0241 | — | — | 7 | 0 |
| MiniLM CE | Top20 | 0.0503 | — | — | 17 | 0 |
| MiniLM CE | matched LLM N | 0.0598 | — | — | 14 | 0 |
| MiniLM CE | score-gap | 0.0425 | — | — | 8 | 0 |

Top-5 / Top-10 仅保留在 diagnostic appendix（m4a21_ranking_metrics.csv / m4a2_k_sensitivity.csv），**不得作为 production variant**。

## 7. Decision Gate（Section 9，参考 LLM F1=0.0664）

- CE matched-N F1=0.0598（LLM=0.0664）；score-gap F1=0.0425（参考 0.0664）。
- **判定：CE_SEMANTIC_CAPACITY_LIMIT**。matched-N 与 score-gap 均未达到 LLM → 下一步才允许测试一次更强 reranker。

## 8. 当前禁止（未违反）

- 未用 Top-5/Top-10 作 production；未据 K-sensitivity 选 K；未调 MiniLM、未 fine-tune；
- 无 score threshold grid search、无 score-gap 参数 tuning、无 CE+RRF fusion、未换 BGE；未改 Planner/Retriever/Prekeep。
- Score-gap 计算仅用 sorted CE scores，0 次读 gold；evaluator 在输出形成后读取。

**本轮（M4A2.1）到此为止：STOP。不自行进入更强 Reranker。**