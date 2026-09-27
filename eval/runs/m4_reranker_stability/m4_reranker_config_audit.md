# M4-0 RERANKER 配置审计（测量口径，不修改）

- 目的：记录当前 LLM Reranker 的全部可调配置，作为稳定性测量的输入口径。
- 只记录，不改任何参数/算法。

## 1. 模型与采样

| 项 | 值 | 备注 |
|---|---|---|
| provider | DeepSeek（openai_base_url=https://api.deepseek.com） | 生产 |
| model | deepseek-chat（llm_model） | tier=strong |
| temperature | 0.0（complete_json 默认） | 非真正确定性 |
| top_p | 未设置（采样随机） | 潜在随机源 |
| seed | 未设置 | 潜在随机源 |
| max_tokens | 1500（rerank batch） | 见 LLMClient |
| 输出格式 | json_object（response_format） | 结构化打分 |

## 2. Reranker Prompt（prompt hash 见搜索实现）

- Prompt 要求逐候选输出 score/label/reason；完整 prompt 文本未在本审计重复（hash 锁定，未修改）。

## 3. 候选分批 / 打分

| 项 | 值 |
|---|---|
| batch_size | 15 |
| max_abstract_chars | 200（abstract 截断） |
| 打分方式 | 逐批串行（serial _score_batch） |
| LLM calls / query | ceil(pool_size / 15) |
| 输出解析 | JSON 解析（完整_json，逐候选 key=paper_id） |

## 4. 截断与排序

| 项 | 值 |
|---|---|
| keep_threshold | 0.35（score>=阈值保留） |
| min_keep | 3（保底） |
| max_results | 20（评测 top_k） |
| tie-break | 按 -score 稳定排序 → 候选插入序 |
| 未覆盖候选 | 词法兜底排末尾 |

## 5. 重试 / 并发 / 状态

| 项 | 值 |
|---|---|
| 重试 | tenacity 3 次，同 payload（无重采样） |
| query 内并发 | 无（串行） |
| 并发顺序改变 | 不影响单 query（串行打分） |
| 随机性来源 | temperature>0 实际采样 + 无 seed/top_p 约束 |

## 6. 结论：唯一随机性来源

LLM 采样本身（temperature 名义 0.0 但非 seed 固定、无 top_p），其余路径全确定性。
同一 pre-rerank 池的 3 次精排输出差异即可归因于 Reranker 采样随机性。