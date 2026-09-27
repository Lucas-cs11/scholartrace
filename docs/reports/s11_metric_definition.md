# S1.1 Gold Lifecycle Metric Definitions

**问题**: S1 report 使用 `raw_gold=14/184`，但该字段实际含义是"LLM Reranker 后 final 输出中命中的 Gold"，与历史 M3-R `raw=25` 的"retrieval-stage raw Gold"语义冲突。

**决策**: 统一使用明确的 lifecycle 术语，禁止继续使用歧义 `raw_gold`。

---

## 标准化术语

### 1. Retrieval Stage
- **`retrieval_raw_unique_gold`**: 召回池中命中的 unique Gold 论文数（去重后，按 paper-level 计数）
- **`retrieval_raw_gold_instances`**: 召回池中 Gold 实例总数（未去重，同一 Gold 多次召回计多次）
- **`retrieval_total_candidates`**: 召回池总候选数
- **`retrieval_deduplicated_candidates`**: 去重后候选数

### 2. Rerank Pool Stage
- **`rerank_pool_unique_gold`**: 进入 LLM Reranker 的 pool 中命中的 unique Gold 数
- **`rerank_pool_size`**: Rerank pool 大小（lexical top-k + assoc safepass）

### 3. Final Output Stage
- **`final_unique_gold`**: LLM Reranker 后最终输出中命中的 unique Gold 数（即用户看到的结果）
- **`final_gold_instances`**: 最终输出中 Gold 实例数（理论上=final_unique_gold，除非 dedup 失败）
- **`final_output_size`**: 最终输出论文数

### 4. Gold Corpus
- **`total_gold_papers`**: Gold corpus 中该查询的 Gold 论文总数（evaluator 口径）
- **`eval_corpus_version`**: 评测集版本标识（hash）

---

## 映射关系

| 旧字段（S1 原始） | 新字段（标准化） | 含义 |
|------------------|-----------------|------|
| `raw_gold` | `final_unique_gold` | 最终输出中命中的 unique Gold |
| `gold_n` | `total_gold_papers` | 该查询 Gold 总数 |
| `retrieved` | `retrieval_total_candidates` | 召回候选总数 |
| `final` | `final_output_size` | 最终输出论文数 |

| M3-R/M5A 历史字段 | 新字段（标准化） | 含义 |
|------------------|-----------------|------|
| `raw` (M3-R=25) | `retrieval_raw_unique_gold` | 召回池 unique Gold |
| `merged` (M5A=27) | `retrieval_raw_unique_gold` (Round1+Round2 merged) | Round-2 后召回池 unique Gold |

---

## Precision / Recall / F1 计算口径

### 当前 S1 口径（final-output-based）
```python
final_unique_gold = len(matched_gold_ids(res.results, gold_groups))
precision = final_unique_gold / final_output_size
recall = final_unique_gold / total_gold_papers
f1 = 2 * precision * recall / (precision + recall)
```

### 历史 M3-R/M5A 口径（retrieval-pool-based）
```python
retrieval_raw_unique_gold = len(matched_gold_ids(retrieval_pool, gold_groups))
recall = retrieval_raw_unique_gold / total_gold_papers
# M3-R/M5A 不计算 precision（因为 pool 太大，precision 无意义）
```

---

## 决策

1. **S1 继续使用 final-output-based P/R/F1**（更接近用户体验）
2. **同时报告 retrieval_raw_unique_gold**（与历史实验可比）
3. **技术论文必须明确区分两种口径**
4. **禁止将 final_unique_gold 标记为 raw_gold**

---

## 实现要求

### run_eval.py 输出字段（per-query）
```python
{
  "query_id": "...",
  # Retrieval stage
  "retrieval_total_candidates": 145,
  "retrieval_deduplicated_candidates": 145,
  "retrieval_raw_unique_gold": 3,  # 新增
  
  # Rerank pool stage
  "rerank_pool_size": 40,  # 新增
  "rerank_pool_unique_gold": 2,  # 新增
  
  # Final output stage
  "final_output_size": 7,
  "final_unique_gold": 1,  # 原 raw_gold
  "final_gold_instances": 1,
  
  # Gold corpus
  "total_gold_papers": 9,  # 原 gold_n
  
  # Metrics
  "precision": 0.1429,  # final_unique_gold / final_output_size
  "recall": 0.1111,     # final_unique_gold / total_gold_papers
  "f1": 0.1250,
}
```

### run_eval.py 输出字段（summary）
```python
{
  "queries_run": 22,
  "ok": 22,
  # Aggregated Gold stats
  "total_retrieval_raw_unique_gold": 25,  # 新增：与 M3-R 可比
  "total_final_unique_gold": 14,          # 原 total_raw_gold
  "total_gold_papers": 184,               # 原 total_gold_n
  
  # Mean metrics（基于 final output）
  "precision": 0.0842,
  "recall": 0.0924,
  "f1": 0.0680,
  
  # Efficiency
  "mean_api_calls": 0.0,
  "mean_llm_calls": 3.4,
  ...
}
```

---

## 验证清单

- [ ] `run_eval.py` 计算 `retrieval_raw_unique_gold` / `rerank_pool_unique_gold` / `final_unique_gold`
- [ ] 去掉所有 `raw_gold` / `gold_n` 字段，替换为标准术语
- [ ] CLI summary 使用标准术语
- [ ] JSON 输出使用标准术语
- [ ] `s1_integration_report.md` 更新指标命名
- [ ] `README.md` 更新指标命名
- [ ] 重新生成 `s1_fast_offline.json` / `s1_deep_offline.json`（新字段）

---

## 与历史实验对齐

| 实验 | retrieval_raw_unique_gold | final_unique_gold | total_gold_papers | 口径 |
|------|--------------------------|-------------------|-------------------|------|
| M3-R | 25 | (未测) | 180 | retrieval-pool |
| M5A Round1 | 25 | (未测) | 180 | retrieval-pool |
| M5A Round2 | 27 | (未测) | 180 | retrieval-pool |
| S1 FAST | (待测) | 14 | 184 | final-output |
| S1 DEEP | (待测) | 15 | 184 | final-output |

待 S1.1 补充 `retrieval_raw_unique_gold` 后，可与 M3-R/M5A 直接对比。

---

**结论**: 
- `raw_gold` 语义污染严重，必须废弃
- 统一使用 `retrieval_raw_unique_gold` / `final_unique_gold` / `total_gold_papers`
- S1 同时报告两种口径，供不同分析场景使用
