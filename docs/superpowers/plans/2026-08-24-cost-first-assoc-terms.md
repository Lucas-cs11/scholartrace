# Cost-First Association Terms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce offline OpenAlex recall cost by limiting planner association terms from six to three while preserving the existing association-candidate reranking behavior and measuring the trade-off on the 22-query difficult subset.

**Architecture:** Keep association terms as planner-generated `SubQuery` objects with `intent=ASSOC_INTENT`; change only their validated count and prompt contract. Increment the planner cache version so old six-term plans are invalidated rather than silently reused. Run the existing benchmark harness against the same 22-query data and caches, using a separate experiment/cache namespace for the three-term A/B result.

**Tech Stack:** Python 3.11+, Pydantic, pytest/pytest-asyncio, OpenAlex adapter, existing `eval_benchmark.py` harness, JSONL plan/recall caches.

---

## Files and responsibilities

- Modify `src/planner.py`: change the association-term contract from six to three and increment `PLANNER_VERSION`.
- Modify `tests/test_planner.py`: update the maximum-term assertion and add coverage that the planner prompt advertises the three-term contract.
- Modify `tests/test_search.py` only if the existing association rerank tests need an explicit regression assertion; no behavior change is expected there.
- Create or modify `scripts/summarize_assoc_ab.py`: compute comparable aggregate metrics and API-call statistics for the six-term baseline and three-term experiment without modifying reports.
- Create or modify `README.md`: document the cost-first offline mode and report the measured A/B numbers after the run; do not claim a full-50 result from the 22-query subset.
- Modify the project memory only after measurements are complete, recording the observed trade-off rather than an unverified expectation.

## Test and experiment isolation

The existing six-term reports under `eval/runs/PASA_ASSOC_*.json` are the baseline and must not be overwritten. The new experiment must use a distinct experiment ID such as `PASA_ASSOC_3` and distinct cache files such as `_pasa_assoc3_recall_cache.jsonl` and `_pasa_assoc3_plan_cache.jsonl`. Because planner cache versioning invalidates old v2 plans, the three-term run must generate fresh plans; recall results may be reused only through a separately named cache and must retain the current source-marking behavior.

### Task 1: Add the failing planner-contract tests

**Files:**
- Modify: `tests/test_planner.py`

- [ ] **Step 1: Update the test import and contract assertion**

Use the existing `MAX_ASSOC_TERMS` import and change the current ten-input test so it asserts exactly three returned association subqueries when ten valid terms are supplied:

```python
llm = FakeLLM(result={"subqueries": [], "assoc_terms": [f"term{i}" for i in range(10)]})
subs = _run(SubQueryPlanner(llm=llm).plan(QueryIR(raw_query="test")))
assoc = [s for s in subs if s.intent == ASSOC_INTENT]
assert len(assoc) == 3
assert len(assoc) == MAX_ASSOC_TERMS
```

- [ ] **Step 2: Add a prompt-contract test**

Add a test that imports `SYSTEM_PROMPT` and asserts the prompt states the new contract, preventing the parser limit and LLM instruction from drifting:

```python
def test_assoc_prompt_requests_three_terms():
    from src.planner import SYSTEM_PROMPT
    assert "恰好 3 个字符串" in SYSTEM_PROMPT
```

- [ ] **Step 3: Run the focused tests before implementation**

Run:

```bash
python3 -m pytest tests/test_planner.py -q
```

Expected: the updated count and prompt tests fail because production still advertises and limits six terms; the existing unrelated planner tests remain passing.

### Task 2: Implement the three-term planner contract

**Files:**
- Modify: `src/planner.py:36-44, 112`

- [ ] **Step 1: Change the prompt and constant**

Replace the prompt phrase `assoc_terms（JSON 数组，恰好 6 个字符串）` with `assoc_terms（JSON 数组，恰好 3 个字符串）`, keep the examples and specificity rules unchanged, and change:

```python
MAX_ASSOC_TERMS = 3
PLANNER_VERSION = 3
```

The version increment is required because version 2 plans contain six-term outputs and must not be loaded into the new planner.

- [ ] **Step 2: Run focused planner and search tests**

Run:

```bash
python3 -m pytest tests/test_planner.py tests/test_search.py -q
```

Expected: all existing planner/search tests pass, including assoc candidate preservation and deduplication.

- [ ] **Step 3: Run the benchmark propagation regression test**

Run:

```bash
python3 -m pytest tests/test_benchmark.py::test_evaluate_reraises_selected_error -q
```

Expected: one pass; this confirms the unrelated quota-stop fix remains intact.

### Task 3: Add deterministic A/B summarization

**Files:**
- Create: `scripts/summarize_assoc_ab.py`

- [ ] **Step 1: Implement report loading and aggregation**

The script must accept two glob patterns and print JSON with report count, mean F1, mean precision, mean recall, positive-F1 count, F1≥0.1 count, F1≥0.2 count, total API calls, and mean API calls. Use only standard library code and fail if either side has a different report count. The core aggregation should be:

```python
def summarize(paths):
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not rows:
        raise ValueError("no reports")
    return {
        "queries": len(rows),
        "mean_f1": round(sum(r["f1"] for r in rows) / len(rows), 4),
        "mean_precision": round(sum(r["precision"] for r in rows) / len(rows), 4),
        "mean_recall": round(sum(r["recall"] for r in rows) / len(rows), 4),
        "f1_positive": sum(r["f1"] > 0 for r in rows),
        "f1_ge_01": sum(r["f1"] >= 0.1 for r in rows),
        "f1_ge_02": sum(r["f1"] >= 0.2 for r in rows),
        "total_api_calls": sum(r["api_calls"] for r in rows),
        "mean_api_calls": round(sum(r["api_calls"] for r in rows) / len(rows), 2),
    }
```

Use `argparse` options `--baseline-glob` and `--candidate-glob`; emit `{"baseline": ..., "candidate": ...}` as indented JSON. Do not import project modules or call APIs.

- [ ] **Step 2: Add a unit test**

Create `tests/test_summarize_assoc_ab.py` with two temporary JSON reports per side and assert the aggregate values, equal-count validation, and empty-glob error. Keep the test entirely offline.

- [ ] **Step 3: Run the script test**

Run:

```bash
python3 -m pytest tests/test_summarize_assoc_ab.py -q
```

Expected: all new tests pass.

### Task 4: Run the isolated three-term A/B benchmark

**Files:**
- Data input: `/tmp/assoc_verify_22.jsonl`
- Baseline reports: `eval/runs/PASA_ASSOC_*.json`
- Candidate reports: `eval/runs/PASA_ASSOC_3_*.json`
- Candidate caches: `eval/runs/_pasa_assoc3_recall_cache.jsonl`, `eval/runs/_pasa_assoc3_plan_cache.jsonl`

- [ ] **Step 1: Confirm the candidate outlet before spending quota**

Run:

```bash
curl -sS --max-time 15 https://api.ipify.org
curl -sS --max-time 15 'https://api.openalex.org/works?search=test&per-page=1' -o /dev/null -w '%{http_code}\n'
```

Proceed only if the HTTP status is `200`; on `429`/`503`, stop and request a new VPN exit.

- [ ] **Step 2: Run the candidate benchmark with isolated names**

Run:

```bash
cd "/Users/lucas/Documents/赛题/scholartrace-contest-backup"
python3 scripts/eval_benchmark.py \
  --data /tmp/assoc_verify_22.jsonl \
  --mode full \
  --recall-source openalex \
  --top-k 20 \
  --experiment PASA_ASSOC_3 \
  --runs eval/runs \
  --cache eval/runs/_pasa_assoc3_recall_cache.jsonl \
  --plan-cache eval/runs/_pasa_assoc3_plan_cache.jsonl \
  --skip-existing
```

If the process exits 42, preserve its caches and stop; do not retry automatically. The run must not touch the baseline reports.

- [ ] **Step 3: Summarize the A/B results**

Run:

```bash
python3 scripts/summarize_assoc_ab.py \
  --baseline-glob 'eval/runs/PASA_ASSOC_RealScholarQuery_*.json' \
  --candidate-glob 'eval/runs/PASA_ASSOC_3_RealScholarQuery_*.json'
```

Expected: both sides contain 22 reports. Report actual numbers only; do not infer a full-50 improvement from this subset.

### Task 5: Document and verify

**Files:**
- Modify: `README.md`
- Modify: project memory file only if the user wants durable experiment history

- [ ] **Step 1: Add a short offline-evaluation note**

Document that `PASA_ASSOC_3` uses at most three association terms, is offline-only, uses isolated caches, and must be compared on the same 22-query difficult subset. Include the measured baseline/candidate values from Task 4 and state whether the cost reduction preserved, improved, or reduced mean F1.

- [ ] **Step 2: Run the complete verification suite**

Run:

```bash
python3 -m pytest -q
python3 -m compileall -q eval scripts src
```

Expected: all tests pass and compilation exits successfully. If async tests fail because the local environment lacks the declared `pytest-asyncio` dependency, install from `requirements.txt` before rerunning; do not change production code for that environment-only failure.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm that baseline reports and caches were not modified or deleted, and that unrelated pre-existing changes in `scripts/eval_after_quota.sh` and `src/adapters/openalex.py` remain untouched.
