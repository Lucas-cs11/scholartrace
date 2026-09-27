# 研迹 ScholarTrace — scholartrace-contest

华为赛题三「科研场景下复杂学术查询的智能论文搜索与推荐」竞赛仓（2026-08-20 建立）。

> 工程原则：赛题优先、接口先于 UI、论文事实不交给 LLM 创造、成本是系统能力。
> 商业/产品侧（engine/ios）与本仓严格隔离，见主开发计划。

## 目录
- `src/schemas.py` — 核心数据对象（QueryIR / SubQuery / PaperIdentity / PaperEvidence / SearchTrace / RankResult / RunReport）
- `src/telemetry.py` — 每次运行的统一成本/延时日志
- `src/adapters/` — 学术搜索 API 适配层（OpenAlex 主召回 / Crossref 身份 / Semantic Scholar 引文）
- `src/llm.py` — LLM 客户端（OpenAI 兼容：DeepSeek，tenacity 重试，telemetry 记账）
- `src/parser.py` — QueryIR 解析（LLM→JSON→pydantic 校验，失败降级为纯 raw_query）
- `src/planner.py` — 子查询分解 + 查询改写（LLM 失败降级为 IR 字段拼凑）
- `src/search.py` — 搜索编排（B0 单查询 / B1 Round0 解析 → Round1 子查询并行召回 → 去重 → 排序 / B2 +S2 引文扩展 / B3 +LLM 精排）
- `api/main.py` — FastAPI 后端层（POST /search mode b0-b3 + GET /health）
- `eval/` — 统一评测 harness + 挑战 query 集（v1 20 条 + v2 草案 20 条）+ 运行记录
- `scripts/` — 运行入口（run_b0/b1/b2/b3.py、curate_gold.py 等）
- `config/settings.py` — pydantic-settings 配置（.env 注入；LLM 用 DeepSeek，OPENALEX_MAILTO 已配置）

## 文档 / 报告
- `proposal/eZScholar_project_proposal.{tex,pdf}` — 项目提案（LaTeX 源 + 编译稿，构建中间产物不入库）
- `docs/reports/` — 各阶段报告（s1_integration、s11\_* 审计、s2\_* 等），脚本生成类报告由 `scripts/s2*_*.py` 自动写入此处
- `docs/` — `claim_evidence_matrix.md`、`failure_limitation_matrix.md` 等结论记录
- `submission/` + `submission.zip` — 竞赛提交包冻结快照（保持原样，勿直接改动；RC 冻结后从主树重建）

## 快速开始
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # LLM: OPENAI_API_KEY/BASE_URL/LLM_MODEL；OpenAlex/Crossref 无需 key
python scripts/run_b0.py     # B0 基线
python scripts/run_b1.py     # B1 查询理解 + 子查询召回
python scripts/run_full.py   # FULL 全链路
python scripts/analyze_runs.py      # S2 跨实验聚合（22 冻结查询严格 A/B，输出 s2_experiments_aggregate.json）
```

## 复现全部评测
```bash
./scripts/reproduce_all.sh   # B0 -> B1 -> B3 -> B4 -> B2 -> FULL（B1 首跑落盘召回缓存，其余复用）
```
> 需要 OpenAlex 配额（~1000 credits/天 ≈ 100 次检索）。摘要写入 `/tmp/b{0,1,2,3,4}.json`、`/tmp/full.json`；
> 逐 query 结果在 `eval/runs/{B0,B1,B2,B3,B4,FULL}_q*.json`。评测无 LLM 时 B0 可直接复现，其余依赖 DeepSeek key。

## 评测进度
| 实验 | 描述 | F1 | 命中/总数 | API calls | latency | 状态 |
|------|------|----|-----------|-----------|---------|------|
| B0 | query→OpenAlex→词法相关度排序 | 0.0095 | 2/20 | 20 | 3.27s | ✅ 完成 |
| B1 | Round0 解析+分解 → 子查询并行召回 → 词法排序 | 待全量 | 部分：q001/q002/q006 R=1.0 | 4-5/query | ~6s | ⏳ 配额恢复后重跑 |
| B3 | B1 召回 → 词法粗筛 → LLM 精排 → 动态截断 | 待全量 | 真实验证：gold 排#1、干扰滤除 | +1-3 LLM/query | — | ⏳ 配额恢复后重跑 |
| B4 | B3 + 召回缓存 + 预算早停（效率分） | 待全量 | — | 缓存命中省 API | — | ⏳ 配额恢复后重跑 |

> B2（S2 引文扩展）、B4（缓存/预算）、API 层均已开发完成，待配额恢复/S2 key 后真实验证。
> B0-B4 阶梯已全部实现；**FULL 全链路已开发**（F1 多轮引文扩展 + F2 早停决策 +
> F3 约束证据链 + F4 search_full 集成，84 单测通过，见 `scripts/run_full.py`）。
>
> 召回失败诊断（2026-08-21）：8 条 B1 召回失败 query 根因是 gold 标题简短专名化
> （"Attention Is All You Need"）与长 AND 子查询词汇不重叠。已修复：planner 新增
> 「标题关键词」子查询策略（简短核心名词单独检索），8 条失败 query 全部验证覆盖 gold
> 标题关键词。B1 全量重跑待 OpenAlex 配额恢复。

> B0 基线结论：冗长自然语言查询直接丢给 OpenAlex 召回率极低（仅 2/20 命中），
> 证明查询改写/子查询分解是提升 F1 的第一杠杆 → B1。
>
> B1 中间结论（已验 6 条）：子查询分解将召回率 0.10 → 1.0（q001/q002/q006）。
> F1 仍卡 0.095 是评测结构所致（单篇 gold 时 top20 的 precision 天花板 0.05）。
>
> B3 思路与验证（2026-08-20）：词法排序在 top20 混入大量不相关候选卡死 precision；
> 改为 LLM 精排 + 动态截断（score≥0.35 保留，保底 3 篇，上限 top_k）。
> 真实 DeepSeek 探针：q001 候选 8 篇（gold PointNet + 5 不相关 + 2 部分相关），
> 精排后 PointNet 第 1（score 1.0/HIGH），不相关全滤除——单篇 gold 时 F1 天花板被突破
> （返回 N 篇则 P=N_gold/N）。全量评测待 OpenAlex 配额恢复后与 B1 一起跑。
>
> ⚠️ 2026-08-20 阻塞：OpenAlex 免费 credits（1000/天/IP）耗尽，429 至次日重置
> （retry-after ≈ 11h）。已加固：并发上限 2、429 退避 5-30s×6、mailto 进 polite pool
> （新 credits 系统下 mailto 不再单独提额，需 help.openalex.org 申请或等重置）。

## 提交包 DoD（核对状态）
| 项 | 状态 |
|----|------|
| 功能覆盖（B0-B4 + FULL 全链路）| ✅ 已实现（97 单测）|
| API 模型注明（OpenAlex/Crossref/S2/OpenCitations/arXiv + DeepSeek）| ✅ README + 适配器注释|
| 可复现评测（reproduce_all.sh + 统一 harness）| ✅ |
| 结构化输出（约束链 + schema_valid 校验）| ✅ |
| 可复现脚本 / 文档 | ✅ README / run_*.py / analyze_runs.py |
| 密钥不入库 | ✅ .env 未跟踪（已审计）|
| RC 冻结 / 唯一提交包 | ⏳ 9/9 前（评测收敛后）|

详见开发计划文档。

---

## S1 参赛系统（FAST / DEEP 统一引擎）

把已验证模块整合成统一比赛引擎（`s1/`），生产冻结 = M3-R（Preserve+Augment）+ LLM Reranker + M5A Planner。
详见 `docs/reports/s1_integration_report.md`。

### 架构
```
FAST:   Round1(Planner→Subqueries→OpenAlex/RecallCache) → Identity Dedup → Lexical+Assoc Pool
        → Production LLM Reranker → Structured Results → SearchTrace
DEEP:   FAST + Round1 Observation(top-8 evidence) → M5A frozen Round2 Planner(follow-ups)
        → Follow-up Retrieval → Merge → Rerank → Results
```
- 结构化输出 `StructuredResult`：metadata（title/author/DOI/year/venue）一律来自学术数据，禁止 LLM 生成。
- `S1SearchTrace`：每次运行完整轨迹（original_question → generated_queries → round1 obs → round2 decision →
  newly_discovered → final_papers + api/llm/tokens/latency）。
- offline 只读 frozen plan/cache，缺失即 `LookupError`（**无 silent fallback**）。
- 不做 Gold-aware trigger；`continue_reason` 仅为 future adaptive trigger 留接口。

### 用法
```bash
# 单问题（FAST 默认；--online 走真实 Planner+OpenAlex）
python3 scripts/run_search.py --question "<Q>" --mode fast --query-id RealScholarQuery_29
python3 scripts/run_search.py --question "<Q>" --mode deep --config configs/deep.yaml --out /tmp/s1_out.json

# 批量评测（默认 offline，22 个冻结查询）
python3 scripts/run_eval.py --mode fast
python3 scripts/run_eval.py --mode deep --out eval/runs/s1/s1_deep_offline.json
python3 scripts/run_eval.py --mode deep --online --limit 2   # 在线冒烟（真实 OpenAlex）

# Q29/Q43 回放 demo（FAST + DEEP 轨迹）
python3 scripts/replay_demo.py

# 自动化测试（离线、无网络/LLM，mock reranker）
python3 -m pytest tests/test_s1_integration.py -q
```
- 配置：`configs/fast.yaml`（max_followup=0、budget=0）、`configs/deep.yaml`（max_followup=3、budget=66）。
- Gold 泄漏：`s1/leakage.py` 静态扫 s1/ 源码 + 运行时 open 审计，杜绝检索代码读 gold。

### 冻结输入（S1 只读）
| 文件 | 用途 |
|------|------|
| `eval/runs/m3r_append/m3r_query_plans.jsonl` | 22 份 M3-R Round1 plan（frozen） |
| `eval/cache/m3r_append/recall_cache.jsonl` | M3-R Round1 召回缓存（338 keys） |
| `eval/runs/m5a_two_round/m5a_round2_plans.jsonl` | M5A Round2 follow-up plans（frozen） |
| `eval/runs/m5a_two_round/m5a_round2_recall_cache.jsonl` | M5A Round2 召回缓存 |
| `eval/runs/m5a_two_round/m5a_plan_meta.json` | M5A planner 版本/prompt_hash |

### S1 实验结果（offline baseline）

**Corpus**: pasa_realscholar_test_b3b570411ce2399c (frozen 22 queries, 184 Gold papers)

| Mode | F1 | final_unique_gold | total_gold_papers | API calls | LLM calls | Status |
|------|-----|-------------------|-------------------|-----------|-----------|--------|
| FAST | 0.0680 | 14 | 184 | 0/q | 3.4/q | ✅ Offline validated |
| DEEP | 0.0641 | 15 | 184 | 0/q | 5.0/q | ✅ Offline validated |

**DEEP Round-2**: executed_queries=62/66, newly_discovered=749, final_round2_papers=144

**指标说明**: `final_unique_gold` 是最终输出命中的 Gold（LLM Reranker 后，用户可见结果）。历史 M3-R `raw=25` 是召回池 Gold（retrieval-stage），两者口径不同。详见 `docs/reports/s11_metric_definition.md`。

**在线验收**: PENDING（LLM balance 不足，详见 `eval/runs/s1/S1_DEFERRED_ACCEPTANCE.md`）

完整结果：`eval/runs/s1/`（`s1_fast_offline.json` / `s1_deep_offline.json`）
