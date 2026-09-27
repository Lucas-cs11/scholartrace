# M4A2 DETERMINISM CHECK

- model: cross-encoder/ms-marco-MiniLM-L6-v2 (max_length=512, device=cpu, inference-only, model.eval())
- sentence-transformers/transformers/torch: sentence_transformers / transformers / torch 2.8.0
- 模型 num_labels=1，activation_fn=Identity → ce_score 为 positive-class 原始 logit（无 sigmoid）；排序只用相对大小，单调一致。
- 同一 frozen pool 完整打分两次（pass1=正式结果，pass2=校验）。

- max |score1 - score2| = 0.00e+00  (tolerance <=1e-6 → PASS)
- rank_order_identical = True
- Top-20 Jaccard = 1.0 (all queries)
- F1 identical (top-20) = True
- 校验 queries = 22

**CROSS-ENCODER DETERMINISM PASSED**（无 CROSS_ENCODER_DETERMINISM_FAILURE）。

## 运行成本
- model load time = 6.5s
- CE inference (2 passes, 22 queries) = 40.5s
- peak RSS = 641 MB (CPU, no VRAM)
- generative LLM reranker calls = 0；OpenAlex physical HTTP = 0