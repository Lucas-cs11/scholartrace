# M4A2_LIGHTWEIGHT_SEMANTIC_RERANKER 决策报告

- 日期：2026-08-30 18:41；模型：cross-encoder/ms-marco-MiniLM-L6-v2（max_length=512，CPU，inference-only）。
- 冻结输入：M3-R_APPEND frozen pool（22 queries）；Planner/Rescue/OpenAlex/Citation/Reference/
  Metadata/Prekeep 全未重跑；CE 排序阶段 0 次读 gold（gold 仅 evaluator）。

## 1. 三方对照（Top-20）

| variant | mean F1 | mean P | mean R | raw | pool | final | final_inst | LLM calls | HTTP |
|---|---|---|---|---|---|---|---|---|---|
| M3-R LLM | 0.0664 | — | — | 25 | 22 | 16 | — | 77 | — |
| M3-R RRF | 0.0241 | — | — | 25 | 22 | 7 | — | 0 | 0 |
| **M4A2 CE** | **0.0503** | 0.0386 | 0.1427 | 25 | 22 | **17** | 17 | 0 | 0 |

- 运行成本：model load 6.5s；CE 打分（2 passes）40.5s；peak RSS=641 MB。

## 2. K 敏感度（diagnostic，不据此选 K）

| K | F1 | P | R |
|---|---|---|---|
| 5 | 0.0719 | 0.0909 | 0.0953 |
| 10 | 0.0708 | 0.0636 | 0.137 |
| 15 | 0.0588 | 0.0485 | 0.1408 |
| 20 | 0.0503 | 0.0386 | 0.1427 |

## 3. Determinism

- max |score1-score2| = 0.00e+00；rank_order_identical=True；Top-20 Jaccard=1.0；F1 identical=True。
- → CROSS-ENCODER DETERMINISM PASSED。

## 4. Gold comparison（M4-0 LOST_RERANKER + M4A1 recovery）

- LOST_RERANKER gold=8；LLM 最终保留=1；RRF top-20 保留=2；CE top-20 保留=5。
- **Q47 FinEval x2**：LLM=[0, 0]，RRF rank=['8', '10']，CE rank=[6, 79]，CE score=[3.81, -6.878]，pre_rerank_rank=[77, 78]。
- **Q15 RLHF Gold**：LLM=[3]，RRF rank=['68']，CE rank=[5]，CE score=[2.925]，pre_rerank_rank=[9]。
- **Q6 systematic FN**：LLM=[0, 0, 0, 0, 0]，RRF rank=['32', '47', '69', '82', '55']，CE rank=[67, 13, 3, 7, 51]，CE score=[-2.481, 2.996, 4.334, 3.705, -0.493]，pre_rerank_rank=[52, 55, 32, 33, 77]。

## 5. Abstract coverage（诊断，不据此调参）

- pool abstract coverage：mean=96.15%（跨 22 queries）。
- gold abstract coverage（池内匹配 gold 组）：mean=95.83%。
- truncated candidates 总数=50；mean CE rank 有 abstract=29.6 vs 无 abstract=55.0 （无 abstract 候选数合计=56）。

## 6. Decision Gate（参考：M3-R LLM F1=0.0664）

- CE F1 = 0.0503（final unique Gold=17），参考 LLM=0.0664。
- **判定：MINILM_CE_INSUFFICIENT**。不调 MiniLM、不 fine-tune、不改阈值；下一步 STOP，由用户决定是否测试更强 reranker。

## 7. 当前禁止（未违反）

- 无 fine-tune、无 Gold 训练、无 hard-negative mining、无多模型 ensemble；
- 无 RRF/LLM/retrieval score fusion、无 safepass bonus、无 handcrafted score；
- 未修改 Planner/Retriever/Safepass/Prekeep；未根据 Gold 选 K；未针对 Q6/Q15/Q47 写特殊规则。

**本轮（M4A2）到此为止：STOP。不自动进入 M4A3。**