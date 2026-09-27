# M4A2.1 DETERMINISM CHECK（M4A2_1_SCORE_GAP）

- Score-gap selection 为纯函数（仅依赖冻结的 sorted CE scores），对每 query 执行两次。
- selected_ids_identical = True
- cardinality_identical = True
- f1_identical = True
- 校验 queries = 22

**SCORE-GAP DETERMINISM PASSED**

## 运行成本
- 全程离线：generative LLM calls = 0；OpenAlex HTTP = 0；CE inference = 0（未加载模型、未重新打分）。