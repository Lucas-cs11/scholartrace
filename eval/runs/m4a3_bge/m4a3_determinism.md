# M4A3 DETERMINISM CHECK

- model: BAAI/bge-reranker-v2-m3（device=cpu，inference-only，model.eval()，params=568M）
- activation_fn=nn.Identity → bge_score 为 positive-class 原始 logit（单调等价，仅用于排序）。
- 同一 frozen pool 完整打分两次（pass1=正式结果，pass2=校验）。

- max |score1 - score2| = 0.000e+00  (tolerance <=1e-6 → PASS)
- rank_order_identical = True
- Top-20 Jaccard = 1.0
- matched-N F1 identical = True
- 校验 queries = 22

**BGE DETERMINISM PASSED**

## 运行成本
- model load = 8.9s（打分进程缓存加载；续跑进程重连 HF 镜像 56.8s）；两遍 inference = 1143.0s + 919.6s
- pairs/s = 1.1；per-query mean latency = 51.95s；peak RSS = 935.1 MB
- generative LLM calls = 0；OpenAlex physical HTTP = 0