# M4-0 RERANKER 稳定性审计 决策报告

- 日期：2026-08-30 14:40；7 条 rich 查询，每查询 3 轮 LLM 精排（相同 pre-rerank 池）
- M3.1 冻结：QUERY_FORMULATION_TUNING_STOP。本审计只测量，不改任何算法。

## 1. 输入冻结证明（Section 3）

7 条 rich 冻结查询重建池 canonical_ids 与落盘快照 rerank_pool.candidate_ids 逐位一致（set+order）。
Planner/OpenAlex/Citation/Reference/Metadata/Prekeep 均未重跑。EXPERIMENT_INVALID 未触发。

## 2. 稳定性证据

- overall mean F1 per run：{1: 0.1348, 2: 0.1245, 3: 0.1266}；range=0.0103
- Top-20 mean Jaccard：mean=0.817 min=0.582
- gold retention flips（3 轮间保留状态翻转的 query-gold 数）=0
- Q15（历史 0.1333→0.0000 污染探针）3 轮 F1=[0.2222, 0.1429, 0.1333]

## 3. Gate（Section 6）

触发随机性判定条件：
  - overall mean F1 range = 0.0103 >= 0.005
  - Top-20 Jaccard 明显不稳定：min=0.582 mean=0.817 (<0.8)

**判定：RERANKER_STOCHASTICITY_CONFIRMED**
**NEXT = M4A_DETERMINISTIC_RERANKER**（不实现，仅记录）

## 4. 零 LLM 损失图谱（Section 7，来自 M3.1 lifecycle）

见 m4_gold_loss_map.csv。LOST_RERANKER（进池但未进 final）候选含 pre_rerank_rank/检索源/safepass/LLM 3 轮选择频率。

**本轮（M4-0）到此为止：只测量。STOP，等待下一步批准，不自动实现 M4A/M4B。**