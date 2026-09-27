# 研迹 ScholarTrace — 赛题提交包

**赛题**：华为赛题三「科研场景下复杂学术查询的智能论文搜索与推荐」
**版本**：S2 Final · 提交版（2026-09-01）
**官方评测语料**：`pasa_realscholar_test_b3b570411ce2399c`（22 条冻结查询，184 篇 Gold）

> 本提交包面向赛题方，包含完整可运行的源码、离线评测数据、S1/S2 结果产物、
> 技术报告与项目提案。开发者向的原始文档见同目录 `DEVELOPER_README.md`。

---

## 一、系统是什么

ScholarTrace 是一套**科研场景复杂学术查询的智能论文搜索与推荐引擎**。它把
「长句自然语言研究查询」转化为可执行的多路子查询检索，经多源召回、引文图谱扩展、
LLM 语义精排，最终输出带**结构化证据链**的论文推荐，并生成**可追溯的检索轨迹**。

设计三原则：

1. **论文事实不交给 LLM 创造** — 标题/作者/年份/期刊/DOI 一律来自学术数据源
   （OpenAlex / Crossref / Semantic Scholar / OpenCitations / arXiv），LLM 仅负责
   查询理解、相关性判断与归纳。
2. **成本是系统能力** — 全链路 telemetry 记账（API 调用 / LLM tokens / 延时），
   提供 FAST / DEEP 两种成本-效果档位。
3. **可复现优先** — 冻结计划与召回缓存支持完全离线重放，评测结果逐字节可复现。

---

## 二、系统架构

```
FAST（默认，低成本）：
  Question → Planner 子查询分解 → OpenAlex 多源召回 → 身份去重
  → 词法 + 联想词精排池 → LLM 语义重排 → 结构化结果 → SearchTrace

DEEP（复杂查询深度模式）：
  FAST Round1 → SearchObservation(证据分析) → M5A 冻结 Round2 Planner
  → 追问检索 → 候选合并 → LLM 重排 → 结构化结果 → SearchTrace
```

**后端 API**（FastAPI）：`POST /search`（模式 b0–b4/full）、`GET /health`、`GET /history`。
**前端**：暗黑学术工作台（`frontend/`），含实时检索管道走廊、遥测仪表板、引文关联图、历史年鉴。

```
FastAPI (api/main.py)
  └─ SearchEngine (src/search.py)          # 搜索编排
       ├─ QueryIRParser (src/parser.py)    # 自然语言 → 结构化约束
       ├─ SubQueryPlanner (src/planner.py) # 子查询分解 + 联想论文名
       ├─ Adapters (src/adapters/)         # OpenAlex/Crossref/S2/OpenCitations/arXiv
       ├─ LLMReranker (src/ranker.py)      # LLM 语义精排 + 证据链
       ├─ SearchSummarizer (src/summarizer.py) # 结果归纳 + 引文图谱
       └─ Telemetry (src/telemetry.py)     # 统一成本/延时记账
```

---

## 三、目录结构

| 路径 | 说明 |
|------|------|
| `src/` | 核心引擎（解析/规划/召回/精排/归纳/遥测 + 观测层） |
| `api/` | FastAPI 后端层 |
| `frontend/` | 前端工作台（HTML/CSS/JS） |
| `config/` + `configs/` | 配置与 FAST/DEEP 模式 YAML |
| `s1/` | S1 参赛统一引擎（FAST/DEEP pipeline） |
| `scripts/` | 运行入口与评测脚本 |
| `tests/` | 自动化测试 |
| `eval/` | 评测 harness + 冻结 Gold + 离线复现缓存 + S1/S2 结果产物 |
| `docs/` | 论证矩阵与失败/局限矩阵 |
| `proposal/` | 项目提案（TeX/PDF） |
| `reports/` | S1/S11/S2 各阶段技术报告 |
| `deploy/` | systemd 部署单元 |

---

## 四、快速开始

### 1. 环境准备

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> 无需任何 API key 即可跑通：OpenAlex / Crossref 免费可用；
> 要启用 LLM 解析/精排/归纳，在 `.env` 填 `OPENAI_API_KEY` + `OPENAI_BASE_URL`；
> 要启用 Semantic Scholar 引文扩展，填 `SEMANTIC_SCHOLAR_API_KEY`。

### 2. 启动后端 + 前端

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# 浏览器打开 http://127.0.0.1:8000
```

### 3. 跑离线评测（无需网络/LLM，冻结数据复现）

```bash
# S1 统一引擎离线评测（FAST / DEEP）
python3 scripts/run_eval.py --mode fast
python3 scripts/run_eval.py --mode deep --out eval/runs/s1/s1_deep_offline.json

# 全量测试（151 用例，mock 网络/LLM）
python3 -m pytest tests/ --ignore=tests/test_analyze.py -q
```

### 4. 复现全部研究评测

```bash
./scripts/reproduce_all.sh   # B0 → B1 → B3 → B4 → B2 → FULL（需 OpenAlex 配额）
```

---

## 五、官方结果（冻结语料）

### S1 离线基准（22 冻结查询 / 184 Gold）

| Mode | F1 | final_unique_gold | total_gold_papers | API calls | LLM calls |
|------|-----|-------------------|-------------------|-----------|-----------|
| FAST | **0.0680** | 14 | 184 | 0/query | 3.4/query |
| DEEP | **0.0641** | 15 | 184 | 0/query | 5.0/query |

DEEP Round-2：executed_queries=62/66，newly_discovered=749，final_round2_papers=144。

### S2 验收

| 门禁 | 状态 |
|------|------|
| CORPUS FROZEN / METRIC SEMANTICS / GOLD LEAKAGE | ✅ PASS |
| FAST / DEEP 结构化输出验证率 | ✅ 100%（233 / 286 篇） |
| 测试套件 | ✅ 通过 |
| 效率评估（FAST 74 LLM 调用、DEEP 109）| ✅ PASS |
| 提交审计（无 .env / 密钥 / 绝对路径 / Gold 运行时泄露）| ✅ PASS |

> 完整产物见 `eval/runs/s1/`、`eval/runs/s2/`；口径定义见 `reports/s11_metric_definition.md`。

---

## 六、评测与结果文件

| 文件 | 说明 |
|------|------|
| `eval/runs/s1/s1_fast_offline.json` | S1 FAST 离线结果 |
| `eval/runs/s1/s1_deep_offline.json` | S1 DEEP 离线结果 |
| `eval/runs/s2/s2_final_metrics_fast.json` | S2 FAST 最终指标 |
| `eval/runs/s2/s2_final_metrics_deep.json` | S2 DEEP 最终指标 |
| `eval/runs/s2/s2_per_query_metrics.csv` | 逐查询指标 |
| `eval/runs/s2/s2_submission_audit.json` | 提交审计 |
| `eval/runs/s2/s2_structured_output_validation.json` | 结构化输出验证 |
| `reports/` | 各阶段技术报告（S1 / S11 / S2） |

---

## 七、安全与合规

- **密钥不入库**：`.env` / `.env.bak` 已被排除，仅保留 `.env.example` 模板。
- **Gold 隔离**：评测 Gold 只进诊断统计，绝不进入检索逻辑（`s1/leakage.py` 静态审计）。
- **无绝对路径**：全部路径相对仓库根目录，可整体移植。
- **论文事实源自学术 API**，LLM 不生成任何事实字段。

---

## 八、技术报告索引

| 报告 | 内容 |
|------|------|
| `reports/s1_integration_report.md` | S1 统一引擎集成 |
| `reports/s11_metric_definition.md` | 官方指标口径定义 |
| `reports/s11_acceptance_report.md` | S1 离线验收 |
| `reports/s2_structured_output_report.md` | S2 结构化输出验证 |
| `reports/s2_efficiency_report.md` | S2 效率评估 |
| `reports/s2_progress.md` | S2 进度与门禁 |
| `docs/claim_evidence_matrix.md` | 论点-证据矩阵 |
| `docs/failure_limitation_matrix.md` | 失败与局限矩阵 |
