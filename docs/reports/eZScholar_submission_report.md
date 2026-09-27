# eZScholar 智能学术搜索 —— 最终提交收口报告

> 生成日期：2026-08-31
> 对象：2026 中国研究生人工智能创新大赛 · 华为赛题三《科研场景下复杂学术查询的智能论文搜索与推荐》
> 口径：所有数字均来自本项目真实运行/审计产物，未编造在线性能。

---

## 一、最终提交包收口结果

### 1.1 提交包审计（`scripts/audit_submission.py`）

当前对工作树跑出的结果（`eval/runs/s2/s2_submission_audit.json`）为 **❌ 尚非全绿，11 项中 9 项通过**：

| 审计项 | 状态 | 说明 |
|---|---|---|
| 无硬编码 API Key | ✅ PASS | 全库扫描未发现明文 Key |
| Gold 不进入生产导入 | ✅ PASS | production 模块不 import eval/gold |
| 无绝对本地路径 | ✅ PASS | **本次已修复**：`eval_benchmark.py` 的 `/tmp/bench_summary.json` 改为仓内相对路径；图脚本的 `/Users/...` skill 路径改为 `NATURE_FIGURE_SKILL` 环境变量注入 |
| 必需配置存在 | ✅ PASS | |
| README 存在 | ✅ PASS | |
| Schema 存在 | ✅ PASS | |
| 语料指纹存在 | ✅ PASS | |
| 无 `.env` | ❌ FAIL | 工作树存在 `.env` / `.env.bak`；**但已 gitignore**（本次新增 `.env.bak`/`.env.*`），均未跟踪，git 提交包不会携带。属"开发机本地密钥、打包即排除"的正常状态 |
| 测试通过 | ❌ FAIL | 见 1.2 |

**结论**：收口差最后两步 —— (a) 打包/CI 时确认排除 `.env*`；(b) 归并未提交的 `analyze_runs.py`/`search.py` 重构以恢复测试全绿。其余安全检查已闭环。

### 1.2 最终测试结果

完整 `pytest` 当前因未提交重构无法一次性全绿，实际状态：

- **可运行测试：152 项 = 150 PASS + 2 FAIL**
  - `tests/test_full.py::test_multi_round_opencitations_enrich`
  - `tests/test_search.py::test_search_b2_opencitations_fallback_enriches_title`
  - 二者为 **mock 单测**（非联网），因未提交的 `src/search.py` OpenCitations 回退补全逻辑改动而断言失配，属进行中重构待归并
- **`tests/test_analyze.py`：1 项收集报错** —— 未提交的 `scripts/analyze_runs.py` 重构移除了 `classify_query / _load_recall_ids / _openalex_keys`，测试未同步更新（git HEAD 版本三者均存在）
- **网络相关测试**：`test_llm.py::test_bad_key_403_passthrough_llm_error` 之前因余额/网络波动失败，**当前已通过**（见第二部分余额已恢复的证据）

> 注：`s11_acceptance_report.md` 另确认 **Gold leakage 15/15 PASS**、SearchTrace/结构化输出 Schema 校验 PASS（均基于 frozen eval）。

### 1.3 复现脚本

- `scripts/reproduce_all.sh` —— 一键复现 B0→B1→B3→B4→B2→FULL，B1 首跑填磁盘召回缓存，后续复用（仅耗 LLM/引文），需 OpenAlex 配额
- `scripts/eval_after_quota.sh` —— 等待 OpenAlex 配额恢复（轮询 HTTP 200）后自动串行 B1→B3→B4→B2（部署机脚本）

### 1.4 提交候选版本

- **Git HEAD**：`f32da06b`（`f32da06b940ac8c2ee0a2c46d22ec1e527a36320`）
- **最近提交**：`f32da06 feat(search): 联想词召回候选保送精排池`、`ef64b83 feat(benchmark): 配额耗尽退出续跑`、`f54a82a feat(planner): 联想词 prompt 加成功示例 25%→41%`
- **Tag**：无（仓库未打 tag）

---

## 二、在线运行验收

### 余额状态：**已恢复，可在线验收**

- **DeepSeek LLM**：✅ 可用（今日 18:15 的 `SMOKE_OA` 冒烟 `llm_calls>0`；Q29/Q43 实时回放 llm=3~5；`test_bad_key_403` 复跑通过）
- **OpenAlex**：✅ 可用（实时探针返回 3 条结果；`s11_acceptance_report.md` 记录的 HTTP 402 余额阻塞已解除）
- 说明：`s11_acceptance_report.md`（2026-08-31）此前标记"PARTIAL — Blocked by LLM API Balance (HTTP 402)"，该阻塞现已解除，在线冒烟已可执行。

### 2.1 / 2.2 FAST 与 DEEP 在线冒烟 + 2.4 Q29/Q43 生产回放

**Q29 / Q43 实时生产回放**（`scripts/replay_demo.py`，真实 DeepSeek + 真实墙钟时延，Q29/Q43 为冻结 22 查询中 evidence-guided Round-2 有增量 Gold 的样例）：

| 场景 | LLM calls | Tokens | 实测时延 (ms) |
|---|---|---|---|
| Q29 FAST | 3 | ≈7,176–7,206 | ≈14,120–16,270 |
| Q29 DEEP | 5 | ≈11,557–11,578 | ≈21,004–22,528 |
| Q43 FAST | 3 | ≈7,057–7,059 | ≈15,238–15,524 |
| Q43 DEEP | 5 | ≈11,763–11,789 | ≈23,248–23,649 |

（两次独立回放给出同一量级；`api=0` 表示该回放未额外计物理 OpenAlex HTTP，LLM 调用与 token 为真实发生。）

### 2.3 真实生产条件下指标

**A. 真实在线冒烟（`run_eval.py --online`，真实 OpenAlex + 真实 DeepSeek，非回放）**

| 指标 | FAST 在线（5 查询） | DEEP 在线（5 查询，部分受限） |
|---|---|---|
| 成功/失败 | **5 / 0** | **2 / 3**（3 条因 OpenAlex 429 限流失败） |
| OpenAlex 逻辑 API calls/query（= 物理 HTTP） | **10.8** | 12.5（仅成功查询口径） |
| LLM calls/query（均值，planner+reranker 合计） | **8.8** | 11.0 |
| Tokens/query（均值） | **18,211** | 24,124 |
| 检索候选数/query（均值） | 148.6 | 164 |
| 最终输出篇数/query（均值） | 9.8 | 13.0 |
| 时延均值 (ms) | **58,239** | ≈79,202 |
| 时延中位 (ms) | **59,356** | — |
| 时延 P90 (ms) | ≈**62,574** | — |
| 时延 P95 (ms) | ≈**62,627** | — |
| F1 / P / R | 0.064 / 0.127 / 0.071 | 0.068 / 0.075 / 0.063 |
| 最终独立命中 Gold | 3 / 63 | 3 / 33（部分） |
| invalid empty title | 0 | 0 |

（P90/P95 基于 5 条样本，为小样本上尾估计。DEEP 在线在并发冒烟中触发 OpenAlex 日配额 429，仅 2/5 完整成功，**其数字仅为参考、不作正式效率结论**；DeepSeek LLM 调用本身正常。）

**B. frozen 评测产物逻辑成本（`s2b_efficiency.py`，22 条查询，物理 HTTP=0 离线回放）**

| 指标 | FAST | DEEP |
|---|---|---|
| LLM calls/query（均值/中位/P90/P95） | 3.4 / 3.0 / 5.8 / 7.0 | 5.0 / 5.0 / 6.7 / 8.7 |
| Tokens/query（均值/中位） | 7,930 / 7,082 | 11,834 / 11,807 |
| 检索候选数（均值） | ≈101 | ≈101 |
| 最终输出篇数（均值） | 10.6 | 13.0 |

> **口径**：A 为真实在线物理时延（含真实 OpenAlex HTTP + DeepSeek），B 为离线回放逻辑成本（physical_http=0）。在线时延显著高于离线回放（~58s vs ~14s），差距来自真实 HTTP 往返与并发上限。

---

## 三、结构化输出完整字段验收

来自 `s2c_structured_output.py`（frozen 22 查询 / 184 Gold）：

| 字段 | FAST (233 结果) | DEEP (286 结果) |
|---|---|---|
| schema 合法率 | 100%（233/233） | 100%（286/286） |
| 缺失 title | 0 | 0 |
| 缺失 authors | **NOT_MEASURED**（frozen artifact 限制） | 同左 |
| 缺失 year | **NOT_MEASURED** | 同左 |
| 缺失 venue | **NOT_MEASURED** | 同左 |
| DOI / OpenAlex ID 完整性 | **NOT_MEASURED** | 同左 |
| invalid DOI 数 | **NOT_MEASURED** | 同左 |
| relevance 解释出现率 | **NOT_MEASURED** | 同左 |

**结论**：`S2-C STRUCTURED_OUTPUT = PASS`（有限校验）。**逐字段完整性（authors/year/venue/DOI/identity/explanation/invalid-DOI）需在线结构化输出审计** —— 在线冒烟产物未保留逐篇结构化字段（row 无 results/papers 键），故这些字段仍无法从可用产物追溯，**如实标注 NOT_MEASURED，不编造**。在线冒烟补充确认 `invalid_empty_title=0`。

**元数据来源策略（S1 设计已确认）**：title/authors/year/venue/DOI/OpenAlex ID 等事实字段一律来自学术数据源，LLM 仅生成 `relevance_explanation / relevance_score / relevance_label`，禁止 LLM 生成事实字段。

---

## 四、最终系统配置（冻结版本）

| 配置项 | 值 |
|---|---|
| Planner 模型 | `deepseek-chat`（LLM_MODEL） |
| Reranker 模型 | `deepseek-chat` |
| 模型 API provider | DeepSeek（`https://api.deepseek.com`） |
| temperature | **0.0**（确定性；`src/llm.py` 默认，未在 .env 覆盖） |
| top_p | 未设置（走 API 默认） |
| max_tokens（分阶段） | planner 1536 / reranker 1500 / parser 1024 / summarizer 1200 / llm 默认 1024 |
| OpenAlex top-k | 20（`top_k=20`，`recall_per_subquery=20`） |
| FAST 最大子查询数 | 5（`B1_MAX_SUBQUERIES=5`，assoc 联想词子查询不受此截断） |
| DEEP follow-up 上限 | 3（`max_followup=3`，`configs/deep.yaml`） |
| Round-2 预算 | 预算上限 66 次新检索；实际执行 62 次（≤66，`ROUND2_BUDGET_ACCOUNTING=PASS`），新增发现论文 749 篇 |
| assoc safepass | **True**（`assoc_safepass=True`，联想词候选保送精排池，不被词法滤掉） |
| 预算默认 | `budget_max_api_calls=50` / `budget_max_rounds=4` / `budget_max_tokens_per_query=20000` |
| Python | 3.13.5 |
| 主要依赖 | pydantic>=2.5 / pydantic-settings / httpx>=0.27 / tenacity>=8.2 / fastapi>=0.110 / uvicorn / pytest / pytest-asyncio |

---

## 五、运行环境

**比赛/部署运行机器（生产部署机）**：
- 操作系统：Ubuntu 22.04（腾讯云）
- 部署路径：`/home/ubuntu/scholartrace-contest`，systemd `scholartrace-api.service`（uvicorn `api.main:app`，host 0.0.0.0，port 8100，1 worker）
- 内存 / 核数 / GPU：详见部署机 `lscpu`/`free`/`nvidia-smi`（本报告未联机采集，不编造；如需请 SSH 采集）

**本地开发机（本次跑图/评测用）**：
- macOS 26.2（Apple M4，10 核，16 GB RAM，无独立 GPU）
- Python 3.13.5

> 在线评测/回放在本机直连 DeepSeek + OpenAlex 完成；部署机承担 API 服务。

---

## 六、多源召回最终决策

**结论：A —— 正式提交版本仍仅使用 OpenAlex，Semantic Scholar 只作为诊断与后续方向。**

依据（配置即证据）：
- `config/settings.py`：`recall_source="openalex"`（默认），`enable_s2_recall=False`（默认关闭）
- S2 多源召回（方向1）为**诊断性探针**（`scripts/s2_d1_multisource_probe.py` 等），结果显示 S2 在相同子查询下可额外带来 Gold（+13 篇 / 7.1%，且与 OpenAlex 不相交），但**未合入正式生产召回路径**

**约束声明**：除非生产代码实际完成并经过正式评测，论文/提案不得将 Semantic Scholar 写成正式生产检索后端。当前论文按此口径表述（Semantic Scholar 仅作为多源召回增益的方向性证据与诊断探针）。

---

## 附：尚未收口的待办

1. 归并未提交重构（`analyze_runs.py`、`search.py` 及受影响测试）→ 恢复 `pytest` 全绿
2. 打包/CI 确认排除 `.env*`（git 已 ignore）
3. DEEP 在线冒烟受 OpenAlex 429 限流仅 2/5 成功，若要完整 DEEP 在线效率需等配额恢复后重跑
4. 若要逐字段结构化审计（authors/year/venue/DOI/explanation），需一次"保留逐篇结构化字段"的在线/离线审计产物
