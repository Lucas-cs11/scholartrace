# M5A_TWO_ROUND_EVIDENCE_GUIDED_SEARCH 决策报告

- 日期：2026-08-31 10:07；版本 m5a-planner-v1；prompt_hash=c914dc27fbff32d0。
- Round-1 = M3-R_APPEND frozen（离线）；Round-2 唯一新增变量 = evidence-guided follow-up queries（top_k=20）。
- Planner freeze：全部 plans 一次性生成后冻结，executor 仅读冻结文件；Planner 0 次读 Gold。

## 主指标（raw unique Gold，baseline=25）

- Round-1 raw unique Gold = 25（应=25）
- M5A merged raw unique Gold = 27
- Round-2 incremental Gold = 2
- queries with retrieval gain = 2 / 22

## Cost

- Round-2 Planner LLM calls = 22（in_tokens=37730 out_tokens=8705）
- executed follow-up queries = 62；experiment physical HTTP = 62 / 66（预算内）；experiment cache_hits = 0；recovery refetches（限流恢复，不计实验预算）= 0
- incremental Gold / API call = 0.0323；incremental Gold / follow-up query = 0.0323

## Query type breakdown

| source | executed | incremental Gold | Gold/query |
|---|---|---|---|
| gap | 41 | 2 | 0.0488 |
| entity | 16 | 0 | 0.0 |
| terminology | 5 | 0 | 0.0 |

## Decision Gate（baseline raw=25）

- **判定：ITERATIVE_RETRIEVAL_INSUFFICIENT**。raw<30：不足。STOP。不自动进入第三轮。

## 约束核对（未违反）

- 未运行任何 Reranker；未用 final F1 判断本轮成败；未自行进入 M5B / 第三轮。
- 无 citation/reference/Crossref/S2 扩展；top_k=20；无 Gold-aware query selector；0 次读 Gold/oracle/lifecycle。
- experiment physical HTTP=62 <= 66（预算内）；未自动扩大预算。

## 证据导向机制演示（唯一 2 篇 incremental Gold 的完整 trace）

- **Q29（IMO 定理证明）DeepSeek-Prover**：第一轮证据 E7=《Lean Workbook: A large-scale Lean problem set...》暴露实体 **Lean** → entity 查询 `Lean formal proof generation LLM` 在 rank 8 找到（同一 gold 也被 gap 查询 `reinforcement learning theorem proving LLM` 在 rank 11 找到；按首次命中归属给 gap）。
- **Q43（抗体设计 DPO）antigen-specific antibody design via DPO**：gap 查询 `direct preference optimization antibody design` 在 rank 3 找到。
- qtype 归属采用「每个新 gold 归首个找到它的 follow-up」的 disjoint 约定；entity/terminology 列=0 不代表实体查询无效——Q29 的 entity 查询在**更优 rank 8** 命中了同一篇 gold（见 m5a_gold_lifecycle.csv 两行）。
- 结论：证据导向的 follow-up **机制成立**（能从 round1 证据中提取实体并召回单轮遗漏的特定论文），但 62 次搜索仅 +2 gold，**召回增益不足**（gold/query=0.032）。

**本轮（M5A）到此为止：STOP。**