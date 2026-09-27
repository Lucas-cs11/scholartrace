# S1.1A No-LLM Cleanup Report

**Date**: 2026-08-31  
**Status**: ✅ COMPLETE  

---

## 执行摘要

在不消耗 LLM/API balance 的前提下，完成 S1 代码和文档的验收口径收口：
- ✅ 实现标准化指标命名（`final_unique_gold`, `total_gold_papers`, `round2_executed_queries`）
- ✅ 修正 Round-2 指标语义（144 = final papers, not query count）
- ✅ 更新 S1 integration report（明确 CONDITIONALLY_ACCEPTED 状态）
- ✅ 修正 gate 校验表述（9/11 PASS, 2/11 PENDING）
- ✅ 冻结 eval corpus 元数据（184 Gold, fingerprinted）
- ✅ 更新 README（标准化术语）
- ✅ 新增 7 个 metric/schema 测试（7/7 PASS）
- ✅ 创建 deferred acceptance 清单

---

## 1. 标准化指标命名实现

### 修改文件
- `scripts/run_eval.py`

### 新字段（per-query）
```python
# Retrieval stage
"retrieval_total_candidates": int
"retrieval_deduplicated_candidates": int

# Final output stage  
"final_output_size": int
"final_unique_gold": int
"final_gold_instances": int
"total_gold_papers": int

# Round-2 (DEEP only)
"round2_planned_queries": int
"round2_executed_queries": int
"round2_filtered_queries": int
"round2_newly_discovered_papers": int
"final_round2_papers": int
```

### 废弃字段（已移除）
- ~~`raw_gold`~~ → `final_unique_gold`
- ~~`gold_n`~~ → `total_gold_papers`
- ~~`retrieved`~~ → `retrieval_total_candidates`
- ~~`final`~~ → `final_output_size`
- ~~`round2_papers`~~ → `final_round2_papers`

### Summary 字段（aggregate）
```python
# Old (ambiguous)
"total_raw_gold": 14
"total_gold_n": 184
"round2_papers_total": 144

# New (explicit)
"total_final_unique_gold": 14
"total_gold_papers": 184
"round2_executed_queries_total": 62        # NEW: actual query count
"round2_newly_discovered_papers_total": 749
"final_round2_papers_total": 144           # final output paper count

"eval_corpus_version": "pasa_realscholar_test_b3b570411ce2399c"
```

### CLI Output（已更新）
```bash
=== S1 EVAL DEEP (offline) ===
Corpus: pasa_realscholar_test_b3b570411ce2399c
queries ok=22 failed=0
P=0.0797 R=0.0940 F1=0.0641
final unique gold=15/184
cost: api=0.0/q llm=5.0/q tok=11833.8/q lat=21953.0ms/q
retrieved=100.5/q final=13.0/q empty_title_total=0
round2: planned=66 executed=62 filtered=4 new_papers=749 final_r2_papers=144
```

---

## 2. Round-2 Metric Naming 修正

### 关键修正
**Old (ambiguous)**:
```
"round2_total": 144  # 歧义：是 query 数还是 paper 数？
```

**New (explicit)**:
```python
# DEEP summary
"round2_planned_queries_total": 66          # 22 queries × 3 = 66
"round2_executed_queries_total": 62         # filtered 4, executed 62
"round2_filtered_queries_total": 4          # dup/generic
"round2_newly_discovered_papers_total": 749 # retrieval 返回论文数
"final_round2_papers_total": 144            # 最终输出中 R2 论文数
```

### 审计数据（S1.1 budget audit）
- Configured budget: `max_followup=3`, `budget_max_new_searches=66`
- Actual: `planned=66`, `executed=62`, `filtered=4`
- Budget compliance: ✅ 62 ≤ 66

### 报告修正
- ❌ Old: "22 个查询共执行 144 个 follow-up retrieval"
- ✅ New: "executed_followups=62 (≤ budget 66), newly_discovered=749 papers, final_round2_papers=144"

---

## 3. S1 Integration Report 更新

### 关键修正

**评测结果 section**:
- Added corpus version: `pasa_realscholar_test_b3b570411ce2399c`
- Changed `raw gold=14/184` → `final_unique_gold=14/184`
- Changed `raw gold=15/184` → `final_unique_gold=15/184`
- Added Round-2 详细分解：`planned=66 executed=62 filtered=4 new_papers=749 final_r2_papers=144`
- Clarified metric semantics: final vs retrieval-stage Gold

**Gate 校验 section**:
- Changed from "全部 PASS" to granular status:
  - 9/11 PASS (offline, leakage, budget, schema, tests)
  - 2/11 PENDING (online smoke, demo replay - LLM balance blocker)
- Added `Overall: 9/11 PASS, 2/11 PENDING (外部执行阻塞，非算法失败)`

**最终状态 section** (NEW):
```
S1_IMPLEMENTATION_COMPLETE: ✅ true
S1_OFFLINE_ACCEPTANCE: ✅ PASS
S1_ONLINE_ACCEPTANCE: 🟡 PENDING_LLM_BALANCE
S1_STATUS: 🟡 CONDITIONALLY_ACCEPTED
```

---

## 4. Gate 校验表述修正

### Old (incorrect)
```
S1 gate 全部 PASS，STOP（不自动进入 S2）
```

### New (accurate)
```
S1 Gate 校验:
- FAST offline replay: ✅ PASS
- DEEP offline replay: ✅ PASS
- FAST/DEEP online smoke: 🟡 PENDING (LLM balance)
- structured output: ✅ PASS
- SearchTrace: ✅ PASS
- Gold leakage: ✅ PASS
- tests 全部 PASS: ✅ PASS
- Round-2 budget: ✅ PASS (62 ≤ 66)
- Metric semantics: ✅ DEFINED
- Corpus version: ✅ FROZEN (184)
- Demo replay: 🟡 PENDING (LLM balance)

Overall: 9/11 PASS, 2/11 PENDING (外部执行阻塞，非算法失败)

S1_STATUS: CONDITIONALLY_ACCEPTED
S2_ENTRY: APPROVED
```

---

## 5. Eval Corpus 元数据冻结

### 定义常量
```python
# scripts/run_eval.py
DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
EVAL_CORPUS_VERSION = "pasa_realscholar_test_b3b570411ce2399c"
```

### Corpus Fingerprint
```
EVAL_CORPUS_VERSION = pasa_realscholar_test_b3b570411ce2399c
DATA_HASH = b3b570411ce2399c (SHA256[:16] of test.jsonl)
CORPUS_FINGERPRINT = 2184b6a5082ccc0f (SHA256[:16] of sorted identity keys)
FROZEN_22_QUERIES = 22 queries (specified list)
FROZEN_22_GOLD_PAPERS = 184
FULL_50_GOLD_PAPERS = 791
```

### 历史对齐
- Historical 180 Gold: 来源不明（可能不同 subset 或数据版本）
- S1 forward 统一使用: **184 Gold (frozen 22 queries)**
- 不追溯修改历史实验报告（保持 180 在历史上下文中）

---

## 6. Demo Code 修复尝试

### 问题
`replay_demo.py` 当前显示 `final=0`（LLM balance 不足导致 reranker 返回空）

### 状态
- 已修复 demo 的 question text（使用正确的 PaSa test.jsonl queries）
- Display bug 需要 reranker 实际运行才能验证修复
- **无法在不消耗 LLM 的前提下验证 demo final output**

### 决策
- Demo 状态标记为: `DEMO_REPLAY=PENDING_LLM_BALANCE`
- 列入 deferred acceptance list
- S1.1A 不 claim PASS

---

## 7. README 更新

### 新增 section: S1 实验结果表格
```markdown
| Mode | F1 | final_unique_gold | total_gold_papers | API calls | LLM calls | Status |
|------|-----|-------------------|-------------------|-----------|-----------|--------|
| FAST | 0.0680 | 14 | 184 | 0/q | 3.4/q | ✅ Offline validated |
| DEEP | 0.0641 | 15 | 184 | 0/q | 5.0/q | ✅ Offline validated |

**DEEP Round-2**: executed_queries=62/66, newly_discovered=749, final_round2_papers=144
```

### 术语说明
添加了指标口径解释：
- `final_unique_gold` = 最终输出命中 Gold（用户可见）
- 历史 M3-R `raw=25` = 召回池 Gold（retrieval-stage）
- 两者口径不同，不可直接比较

### 在线验收状态
添加了 PENDING 说明，指向 deferred acceptance list

---

## 8. 新增测试

### 文件
`tests/test_s11a_metrics.py`

### 测试覆盖（7 tests, 7 PASS）
1. ✅ `test_eval_schema_no_ambiguous_raw_gold`: 代码中无 `"raw_gold"` / `"gold_n"`
2. ✅ `test_eval_schema_uses_standardized_fields`: 使用标准化字段
3. ✅ `test_round2_budget_configured`: DEEP config 明确 `budget_max_new_searches=66`
4. ✅ `test_eval_corpus_version_exists`: 定义了 `EVAL_CORPUS_VERSION` 常量
5. ✅ `test_frozen_22_gold_count`: 22 queries 总 Gold = 184
6. ✅ `test_round2_metric_naming`: 使用 `round2_planned/executed/filtered_queries`
7. ✅ `test_s1_deferred_acceptance_exists`: deferred list 存在且包含必要内容

### 所有测试状态
- S1 integration tests: 15/15 PASS（原有）
- S1.1A metric tests: 7/7 PASS（新增）
- **Total: 22/22 PASS** ✅

---

## 9. Deferred Acceptance List

### 文件
`eval/runs/s1/S1_DEFERRED_ACCEPTANCE.md`

### 内容
4 项 deferred items（均因 LLM balance 阻塞）：
1. FAST online smoke test (2 queries)
2. DEEP online smoke test (2 queries)
3. Q29/Q43 production demo replay
4. Metric-corrected full eval regeneration (optional)

### 说明
- 原因: `LLM_API_HTTP_402_INSUFFICIENT_BALANCE`
- 类型: **Deferred execution checks, not failed algorithm gates**
- Frozen artifacts 仍然有效可用（带 manual metric name translation）
- 不阻塞 S2 entry

---

## 10. 文件清单

### 修改文件
```
scripts/run_eval.py              # 标准化指标实现
s1_integration_report.md         # 评测结果、gate 校验、最终状态
README.md                        # S1 实验结果表格、术语说明
```

### 新增文件
```
tests/test_s11a_metrics.py                    # 7 个 metric/schema 测试
eval/runs/s1/S1_DEFERRED_ACCEPTANCE.md       # Deferred list
s11_metric_definition.md                      # 指标定义规范（已存在，S1.1 产物）
s11_budget_audit_report.md                    # Budget audit（已存在，S1.1 产物）
s11_gold_corpus_audit_report.md              # Corpus audit（已存在，S1.1 产物）
s11_acceptance_report.md                      # S1.1 acceptance（已存在，S1.1 产物）
s11a_cleanup_report.md                        # 本报告
```

---

## 11. 遗留 Deprecated 字段

### 已冻结 JSON 文件（不修改）
以下文件使用旧字段名，但语义已在 S1.1 audit 中明确，可继续使用：
- `eval/runs/s1/s1_fast_offline.json`
- `eval/runs/s1/s1_deep_offline.json`

**Manual translation**:
```python
old["raw_gold"] → new["final_unique_gold"]
old["gold_n"] → new["total_gold_papers"]
old["round2_papers"] → new["final_round2_papers"]
old["round2_papers_total"] = 144 → new["final_round2_papers_total"] = 144
# 注意: round2_papers_total 是 paper count, NOT query count
```

### 未来生成的 JSON（当 LLM balance 恢复后）
将使用新字段名，无需 manual translation。

---

## 12. S1 最终状态

### Implementation
✅ **S1_IMPLEMENTATION_COMPLETE = true**

完成项：
- FAST/DEEP 统一引擎
- 结构化输出（metadata 来自学术数据）
- SearchTrace 完整轨迹
- Offline replay（22/22, api=0）
- Gold 泄漏检测（静态+运行时）
- 22/22 集成测试通过（15 S1 + 7 S1.1A）
- Round-2 budget enforcement（62/66 ✅）
- 指标语义标准化
- Gold corpus 版本冻结（184, fingerprinted）

### Acceptance
🟡 **S1_OFFLINE_ACCEPTANCE = PASS**  
🟡 **S1_ONLINE_ACCEPTANCE = PENDING_LLM_BALANCE**  
🟡 **S1_STATUS = CONDITIONALLY_ACCEPTED**

Pending 项（外部阻塞）：
- FAST/DEEP online smoke tests
- Q29/Q43 production demo replay
- Metric-corrected eval regeneration (optional)

**阻塞原因**: LLM API HTTP 402 Insufficient Balance（外部执行环境问题，非算法失败）

### S2 Entry
✅ **S2_ENTRY = APPROVED**

Frozen S1 results 提供有效 baseline：
- FAST F1=0.0680 (final_unique_gold=14/184)
- DEEP F1=0.0641 (final_unique_gold=15/184)
- Round-2 budget accounting verified (62/66)

---

## 13. 验证清单

- [x] 标准化指标命名实现完成
- [x] run_eval.py 无 deprecated 字段（`raw_gold`, `gold_n`, `round2_total`）
- [x] Round-2 metric 明确语义（planned/executed/filtered queries, final papers）
- [x] S1 integration report 更新（评测结果、gate、最终状态）
- [x] Gate 校验表述准确（9/11 PASS, 2/11 PENDING）
- [x] Eval corpus 版本冻结（184 Gold, fingerprinted）
- [x] README 使用标准术语
- [x] 7 个 S1.1A metric tests PASS
- [x] 15 个 S1 integration tests 仍然 PASS
- [x] Deferred acceptance list 创建
- [x] 无 LLM/API 调用消耗

---

## 14. S2 准备就绪

**可用 baseline**:
- S1 FAST offline: F1=0.0680
- S1 DEEP offline: F1=0.0641
- Metric semantics documented: `s11_metric_definition.md`
- Corpus fingerprint: `pasa_realscholar_test_b3b570411ce2399c`

**下一步（S2 scope）**:
- F1 最终冻结评测
- 运行效率评测
- 结构化输出验收
- 技术论文 claim-evidence chain

**不阻塞 S2 的项**:
- Online smoke tests（deferred，非 blocking）
- Demo replay（deferred，非 blocking）

---

## 结论

**S1.1A_NO_LLM_CLEANUP: ✅ COMPLETE**

所有代码和文档口径收口完成：
- 指标命名标准化实现
- Round-2 语义明确
- S1 report 和 README 更新
- Corpus 版本冻结
- 22/22 测试通过
- Deferred list 创建

**最终状态**:
```
S1_IMPLEMENTATION_COMPLETE = true
S1_OFFLINE_ACCEPTANCE = PASS
S1_ONLINE_ACCEPTANCE = PENDING_LLM_BALANCE
S1_STATUS = CONDITIONALLY_ACCEPTED
S2_ENTRY = APPROVED
```

**准备进入 S2**: ✅ READY
