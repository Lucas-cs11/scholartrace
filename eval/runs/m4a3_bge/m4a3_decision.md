# M4A3_STRONGER_RERANKER 决策报告

- 日期：2026-08-30 22:03；模型：BAAI/bge-reranker-v2-m3（inference-only, eval, CPU）。
- 上游完全冻结：M3-R_APPEND frozen pool（raw=25, pool=22）；Planner/Rescue/OpenAlex/Citation/Reference/Metadata/Prekeep 未重跑。
- 池一致性：22/22 set+order 与 M3-R rerank_pool snapshot 逐位一致 → EXPERIMENT_INVALID 未触发。
- 输入：query=原始问题；passage=title+"\n"+abstract（缺 abstract 仅 title）；max_length=512=模型官方推荐值；0 次读 Gold/Oracle/历史 loss map。

- 主比较（matched-N，N_q=LLM final count）：mean F1=0.0588，final unique Gold=15，instances=15。
- Top-20（辅助）：mean F1=0.0496。
- Ranking metrics：MRR@20=0.1868 MAP@20=0.0603 R@5=0.0979 R@10=0.1067 R@20=0.1765 NDCG@5=0.1148 NDCG@10=0.1077 NDCG@20=0.1235。

## 对照表（vs LLM / MiniLM / RRF）

| variant | selection | F1 | final Gold | MRR@20 | MAP@20 | NDCG@5 | gen LLM calls |
|---|---|---|---|---|---|---|---|
| M3-R LLM | variable-N (threshold) | 0.0664 | 16 | 0.369 | 0.12 | 0.207 | 77 |
| RRF | Top20 | 0.0241 | 7 |  |  |  | 0 |
| MiniLM CE | matched LLM N | 0.0598 | 14 | 0.184 | 0.051 | 0.099 | 0 |
| BGE v2-m3 | matched LLM N | 0.0588 | 15 | 0.1868 | 0.0603 | 0.1148 | 0 |
| BGE v2-m3 | Top20 (辅助) | 0.0496 | 16 |  |  |  | 0 |

## Determinism

- det_ok=True（max|Δscore| / rank / Jaccard / F1 详见 m4a3_determinism.md）。

## 系统性 FN 对照（Q6×3/5、Q15 RLHF、Q47 FinEval ×2，只解释不改模型）

| query | gold | pre_rank | LLM rank | LLM sel | RRF rank | MiniLM rank | BGE rank | BGE sel |
|---|---|---|---|---|---|---|---|---|
| RealScholarQuery_15 | understandingtheeffectso | 9 | 8 | 1 | 68 | 5 | 2 | 1 |
| RealScholarQuery_47 | finevalachinesefinancial | 78 | 0 | 0 | 10 |  | 80 | 0 |
| RealScholarQuery_47 | finevalachinesefinancial | 77 | 0 | 0 | 8 | 6 | 3 | 1 |
| RealScholarQuery_6 | imragmultiroundretrieval | 32 | 0 | 0 | 69 | 3 | 3 | 1 |
| RealScholarQuery_6 | learningtodecomposehypot | 77 | 0 | 0 | 55 |  | 30 | 0 |
| RealScholarQuery_6 | answeringquestionsbymeta | 52 | 0 | 0 | 32 |  | 52 | 0 |
| RealScholarQuery_6 | recitationaugmentedlangu | 33 | 0 | 0 | 82 | 7 | 8 | 1 |
| RealScholarQuery_6 | raftadaptinglanguagemode | 55 | 0 | 0 | 47 | 13 | 21 | 0 |

## Efficiency

- model load=8.9s（打分进程缓存加载，run.log 实测；续跑进程重连 HF 镜像 56.8s）；pass1 infer=1143.0s（pass2=919.6s）；pairs/s=1.1；per-query=51.95s；peak RSS=935.1MB（续跑进程；打分进程 ps 观测 ≈1.5GB）。
- generative LLM calls = 0；OpenAlex physical HTTP = 0。

## Decision Gate（参考 LLM matched-N F1=0.0664）

- BGE matched-N F1=0.0588（LLM=0.0664，MiniLM=0.0598）。
- **判定：DETERMINISTIC_RERANKER_QUALITY_LIMIT**。BGE 未达 LLM 且 < 0.060。不再测试其他 reranker。production 暂时保留 M3-R LLM Reranker。

## RERANKER_RESEARCH_FREEZE

- 本轮后冻结 M4：禁止自动进入第三个 reranker / ensemble / fine-tuning / threshold search / K tuning / LLM voting / CE+RRF fusion。
- 下一研发主线转向：Iterative / Agentic Retrieval。

**本轮（M4A3）到此为止：STOP。总耗时 78s。**