# M3.1 Prompt Audit（零联网，仅 query 文本 + 既有 evaluator 结果）

日期：2026-08-27。范围：15 条 sparse question + 已冻结的 M3 rescue plans（`eval/runs/m3_sparse_plan_rescue/m3_sparse_plans.jsonl`）。
只分析 query 文本本身，不进行任何新检索；不读取 Gold title/author/DOI 来设计 Prompt。
历史 per-type incremental Gold 来自 M3-R 已落盘的 `m3r_gold_lifecycle.csv`（evaluator 结果，非 Prompt 输入）。

## 1. 结构统计（15 条 sparse，M3 已冻结 Rescue）

| 指标 | 值 |
|---|---|
| CORE 数量 | 15（全部 15 条都生成了 core） |
| ANCHOR 数量 | 24 |
| DISCOVERY 数量 | 15 |
| Rescue 子查询总数 | 54（avg 3.6 / query；上限 4） |
| 与 original 高度重复（Jaccard≥0.7） | 1（仅 RealScholarQuery_48） |
| 明显过泛 query（仅领域词、无 task/method/entity） | 0（现有 filter 已拦截） |
| 缺少 task/method/entity 组合（单概念） | 0（现有 filter 已拦截） |
| 不同 Rescue query 之间高度重复 | 0（Jaccard 层面无近义） |

## 2. 结构结论：过滤层健康，但检索假设高度冗余

结构过滤（exact/norm dedup + generic + single-concept）把「低质」query 挡在门外，所以
generic/single/dup 全为 0。但**语义层面 54 条 query 大量是同一检索假设的换词改写**，例如：

- Q38：`image encoding distributions` / `visual encoding distribution` / `image representation distribution` —— 同一假设改三个词。
- Q20/Q21/Q39：多为 `LLM + <任务>` 的重新组合，CORE 与 ANCHOR 只是换个词序。
- CORE 普遍是把原问题压缩成 `LLM + 领域`，缺少独立的方法/基准/实体锚点。

**诊断：M3 的 Rescue 属于「natural-language synonym rewriting」，不是「互补的学术检索假设」。**
这正是 M3.1 Prompt 要改的：让 CORE/ANCHOR/DISCOVERY 各自承担**不同的检索假设**（不同
benchmark/dataset/method/entity × task 组合），而不是同一概念的不同措辞。

## 3. 既有 M3-R 各 query type 的历史 incremental Gold（evaluator 结果）

| query_type | 生成数 | incremental unique Gold | incremental Gold/query | 示例命中 |
|---|---|---|---|---|
| CORE | 15 | 0 | 0.000 | — |
| **ANCHOR** | 24 | **2** | 0.083 | cruxeval(benchmark,Q28)、adaframe(method,Q42) |
| DISCOVERY | 15 | 1 | 0.067 | LLM few-shot reranker(domain bridge,Q17) |

**结论：ANCHOR 是唯一有效产出（2/3），且都来自「specific entity + task」组合**
（`cruxeval a benchmark for code reasoning`、`adaframe adaptive frame selection for fast video recognition`）。
CORE（压缩原句换词）贡献 0；DISCOVERY（词汇桥）贡献 1。

## 4. M3.1 Prompt 设计依据（从 query 文本结构推导，不读 Gold）

1. **CORE 必须从「改写原句」改为「压缩研究问题为高密度名词短语」**：保留 task + method/domain，
   去掉自然语言请求成分（papers about / research on / show me / methods for）。CORE 目前是纯换词 → 改规则。
2. **ANCHOR 是本轮重点**：强制「specific anchor（benchmark/dataset/model/method/专名概念）+ task/method/context」
   组合；question 中有明确实体时必须产出含该实体的 ANCHOR；无实体时不得编造，改用具体 task/method 组合。
   已有证据 ANCHOR 是 workhorse（2/3 incremental）→ 提高其假设多样性。
3. **DISCOVERY 从「普通同义改写」改为解决 VOCABULARY_MISMATCH + COMPOSITIONAL_QUERY_FAILURE**：
   让 planner 思考「论文作者会用哪个学术术语描述这个研究范式」，允许方法类别/社区任务名/标准术语/
   两关键概念重组；禁止只换一两个普通英文词形成伪多样性。
4. **Diversity Gate 增强**：新增「与 original 几乎等价 → drop」「两个 rescue 核心 token 高度重合且检索假设相同
   → 保留更具体者」。Rescue 数量允许 2~4，不凑满 4。

## 5. 范围声明

- 零联网；未读取任何 Gold title/author/DOI/abstract 来设计 Prompt。
- per-type incremental Gold 仅作为 evaluator 反馈（哪类 query 有效），未进入 Production Planner 输入。
- Prompt 修订为统一规则，不针对某一条测试 query 手工优化。
