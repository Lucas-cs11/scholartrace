"""M3.1_PLANNER_PROMPT_REVISION：唯一一次 Query Formulation Prompt 修订（M3_1_PLANNER_VERSION=2）。

用户批准 M3-R_APPEND（raw=25，QUERY_FORMULATION_PARTIAL_SIGNAL）后，本轮做一次且仅一次
Prompt Revision：让 Rescue Query 从「自然语言同义改写」变成「互补的学术检索假设」。
本轮结束（无论结果）后 STOP Query Formulation Prompt tuning；不自动进入 M4。

对照：B0 = PASA_ASSOC_NO_CIT（F1=0.0618, R=0.1339, raw=22）；M3-R_APPEND（raw=25, F1=0.0664）。
本轮唯一变量 vs M3-R：SparsePlanRescue Prompt（M3_1_PLANNER_VERSION=2 + SYSTEM_PROMPT_M31）。

冻结清单（一条不超）：不重构 Planner、不增加 query 数量、不改变 Preserve+Augment；
assoc_safepass=True（KEEP_SAFEPASS）、Citation/Reference/Metadata OFF、OpenAlex 不变、
top_k=20、Prekeep 不变、Reranker 不变；7 条 rich v2 plan 原样。
禁止：M4 reranker opt、Adaptive Retrieval、Multi-provider、Early Stop、top-k 改、citation restore、
Quality Selector、per-Gold 调优。

APPEND 语义（同 M3-R）：15 条 sparse plan 的 final_search_queries =
    original_v2_subqueries + m31_rescue_subqueries（追加，Preserve + Augment）。

缓存/网络预算（Step 10）：
- original/rich 子查询读 _pasa_recall_cache（0 联网）。
- M3.1 rescue 子查询若与 M3 已缓存子查询同文本 → 从 M3 response cache 重放（0 联网）。
- 仅【新 M3.1 rescue 子查询】（不在任何缓存）请求 OpenAlex。
- 新增 OpenAlex search <= 60；production-equivalent 总 logical searches <= 150。
  任一超限 → STOP，不自动削减 query、不自动执行。

Gold isolation（Step 9，EXPERIMENT_INVALID 门禁）：Production Planner 只喂 question 文本；
绝不接触 gold title/author/DOI/arXiv/abstract/Oracle probe/per-Gold taxonomy。gold 只在
evaluator 于 retrieval 完成后读取。发现泄漏 → EXPERIMENT_INVALID=true。

产物（eval/runs/m31/）：
  m31_prompt_audit.md        —— 零联网 Prompt 审计（Step 2，已落盘）
  m31_plans.jsonl            —— 15 条 M3.1 rescue plan（含 version/prompt_hash/plan_hash/raw_counts）
  m31_query_plans.jsonl      —— 22 条统一 plan cache（7 rich 冻结 + 15 sparse append）
  m31_query_type_metrics.csv —— 每 sparse query CORE/ANCHOR/DISCOVERY 生成/执行/增量/裁剪
  m31_gold_lifecycle.csv     —— 全 gold 生命周期（含 source_kind/query_type/append）
  m31_vs_b0_vs_m3r.csv       —— B0 vs M3-R vs M3.1 三方（逐 query + MEAN）
  m31_cost_breakdown.csv     —— 生产等价 logical / 物理 HTTP / 复用 三方成本
  m31_decision.md            —— 决策（Step 12 gate + Step 13 F1 判定）

用法：
  python scripts/run_m31.py generate        # 重新生成 15 条 M3.1 rescue plan（LLM）
  python scripts/run_m31.py build           # 组合 rich(冻结)+append(rescue) -> plan cache（离线）
  python scripts/run_m31.py search          # 22 条统一评测（预算化联网，先验配额）
  python scripts/run_m31.py report          # 七份产物 + 决策（离线）
  python scripts/run_m31.py all
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from eval.harness import compute_p_r_f1, match_gold
from src.adapters.openalex import OpenAlexAdapter
from src.observability.canonical import is_gold_by_title, norm_title
from src.observability.response_cache import CacheMiss, ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.planner import ASSOC_INTENT
from src.planner_rescue import (
    INTENT_ANCHOR, INTENT_CORE, INTENT_DISCOVERY, M3_1_PLANNER_VERSION,
    SparsePlanRescue, SYSTEM_PROMPT_M31,
)
from src.schemas import PaperEvidence
from src.search import SearchEngine
from scripts.eval_benchmark import load_pasa

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
V2_PLAN_CACHE = "eval/runs/_pasa_plan_cache.jsonl"
NOCIT_RESULTS = "eval/runs/nocitation/nocitation_results.json"
RECALL_CACHE = "eval/runs/_pasa_recall_cache.jsonl"
M3_CACHE_DIR = "eval/cache/m3_sparse_plan_rescue"  # M3 本轮生成的 rescue response cache（复用）
M3R_RUN_DIR = Path("eval/runs/m3r_append")          # M3-R reports + traces
M3R_TRACE_DIR = Path("eval/diagnostics/m3r_append")
M31_DIR = Path("eval/runs/m31")
RESCUE_PLAN_FILE = M31_DIR / "m31_plans.jsonl"
META_FILE = M31_DIR / "m31_plan_meta.json"
PLAN_FILE = M31_DIR / "m31_query_plans.jsonl"
CACHE_DIR = "eval/cache/m31"                          # M3.1 新联网响应 cache（本 run 新写入）
TRACE_DIR = Path("eval/diagnostics/m31")
AUDIT_FILE = M31_DIR / "m31_prompt_audit.md"
EXPERIMENT = "PASA_ASSOC_M31_PLANNER_REV"
BASELINE_REF = "PASA_ASSOC_NO_CIT"
M3R_REF = "PASA_ASSOC_M3R_APPEND"
TOP_K = 20
RESCUE_INTENTS = (INTENT_CORE, INTENT_ANCHOR, INTENT_DISCOVERY)
NEW_NETWORK_BUDGET = 60     # Step 10：新增 OpenAlex search <= 60
PROD_EQUIV_BUDGET = 150     # Step 10：production-equivalent 总 logical searches <= 150
QUOTA_MARGIN = 20
OPENALEX_PROBE = "https://api.openalex.org/works?search=test&per-page=1"


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_v2_plans(path: str) -> dict[str, dict]:
    plans: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("v") != 2 or not isinstance(d.get("ir"), dict) or not isinstance(d.get("subs"), list):
            raise ValueError(f"v2 冻结 plan 非法: {d.get('query')}")
        plans[d["query"]] = d
    return plans


def load_rescue_plans() -> dict[str, dict]:
    plans: dict[str, dict] = {}
    if not RESCUE_PLAN_FILE.exists():
        return plans
    for line in RESCUE_PLAN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        plans[d["query"]] = d
    return plans


def load_query_plans() -> dict[str, dict]:
    plans: dict[str, dict] = {}
    if not PLAN_FILE.exists():
        return plans
    for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        plans[d["query"]] = d
    return plans


def load_recall_cache_map(path: str = RECALL_CACHE) -> dict[str, list]:
    m: dict[str, list] = {}
    if not Path(path).exists():
        return m
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        m[d["q"]] = [PaperEvidence(**e) for e in d["evs"]]
    return m


def load_nocit_baseline() -> dict[str, dict]:
    d = json.loads(Path(NOCIT_RESULTS).read_text(encoding="utf-8"))
    assert d["experiment"] == BASELINE_REF, f"基线实验名不符: {d['experiment']}"
    return {r["query_id"]: r for r in d["results"]}


def select_queries() -> list[dict]:
    b0 = load_nocit_baseline()
    v2 = load_v2_plans(V2_PLAN_CACHE)
    all_q = load_pasa(DATA)
    picked = [q for q in all_q if q["query_id"] in b0 and q["query"] in v2]
    if len(picked) != len(b0):
        raise SystemExit(f"EXPERIMENT_INVALID: 过滤后 {len(picked)} 条 ≠ B0 基线 {len(b0)} 条")
    return picked


def query_id_map() -> dict[str, str]:
    return {q["query"]: q["query_id"] for q in load_pasa(DATA)}


def classify_sparse_rich(v2: dict[str, dict]) -> tuple[set[str], set[str]]:
    sparse: set[str] = set()
    rich: set[str] = set()
    for q, d in v2.items():
        subs = d.get("subs", [])
        assoc = sum(1 for s in subs if s.get("intent") == ASSOC_INTENT)
        if assoc == 0 or len(subs) <= 1:
            sparse.add(q)
        else:
            rich.add(q)
    return sparse, rich


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM_PROMPT_M31.encode("utf-8")).hexdigest()


def plan_hash_of(plans: list[dict]) -> str:
    canon = {"m31_planner_version": M3_1_PLANNER_VERSION,
             "prompt_hash": prompt_hash(),
             "plans": [{**{k: p[k] for k in ("query", "subs")}} for p in sorted(plans, key=lambda x: x["query"])]}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------
# Phase A：generate —— 15 条 sparse 的 M3.1 rescue plan（LLM，一次 / query）
# 所有 15 条用同一 prompt（SYSTEM_PROMPT_M31 / M3_1_PLANNER_VERSION=2），冻结 version+prompt_hash+plan_hash。
# --------------------------------------------------------------------------
async def cmd_generate(limit: int | None) -> None:
    M31_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_v2_plans(V2_PLAN_CACHE)
    sparse, rich = classify_sparse_rich(v2)
    existing = load_rescue_plans()
    todo = [q for q in sorted(sparse, key=lambda x: query_id_map()[x]) if q not in existing]
    if limit:
        todo = todo[:limit]
    print(f"M3.1 rescue plan 生成：sparse {len(sparse)} 条（rich {len(rich)} 冻结不动），"
          f"已存在 {len(existing)} 条，本次生成 {len(todo)} 条")
    planner = SparsePlanRescue()  # 默认 M3_1_PLANNER_VERSION=2 + SYSTEM_PROMPT_M31
    for i, q in enumerate(todo, 1):
        t0 = time.time()
        subs = await planner.plan_raw(q)
        raw_counts = getattr(planner, "last_raw_counts", {})
        plan = {
            "query": q,
            "query_id": query_id_map()[q],
            "m31_planner_version": M3_1_PLANNER_VERSION,
            "prompt_hash": prompt_hash(),
            "raw_counts": raw_counts,          # LLM 原始输出各 intent 数量（generated 口径）
            "subs": [s.model_dump() for s in subs],  # 执行口径（Diversity Gate 后）
        }
        plan["plan_hash"] = hashlib.sha256(
            json.dumps({"v": M3_1_PLANNER_VERSION, "subs": plan["subs"]},
                       sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with open(RESCUE_PLAN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
        tag = " ".join(f"{s['intent']}={s['query_text'][:26]}" for s in plan["subs"])
        print(f"[{i}/{len(todo)}] {plan['query_id']} ({round((time.time() - t0) * 1000)}ms) {tag}")
    all_rescue = sorted(load_rescue_plans().values(), key=lambda x: x["query"])
    META_FILE.write_text(json.dumps({
        "m31_planner_version": M3_1_PLANNER_VERSION,
        "prompt_hash": prompt_hash(),
        "plan_hash": plan_hash_of(all_rescue),
        "sparse_count": len(sparse), "rich_count": len(rich),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    cnt = Counter()
    for p in all_rescue:
        cnt.update(s["intent"] for s in p["subs"])
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    print("M3.1 rescue plan 类型统计（执行后）：", dict(cnt))
    print(f"M3_1_PLANNER_VERSION={M3_1_PLANNER_VERSION}  prompt_hash={meta['prompt_hash'][:16]}…  "
          f"plan_hash={meta['plan_hash'][:16]}…")
    print("写", RESCUE_PLAN_FILE)


# --------------------------------------------------------------------------
# Phase B：build —— rich(冻结原样) + sparse(Preserve+Augment: original v2 + m31 rescue) -> plan cache
# --------------------------------------------------------------------------
def cmd_build() -> None:
    M31_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_v2_plans(V2_PLAN_CACHE)
    rescue = load_rescue_plans()
    sparse, rich = classify_sparse_rich(v2)
    missing = sparse - set(rescue)
    if missing:
        raise SystemExit(f"EXPERIMENT_INVALID: 缺 {len(missing)} 条 rescue plan，先跑 generate：{sorted(missing)[:3]}…")
    queries = select_queries()
    lines = []
    for q in queries:
        if q["query"] in rich:
            lines.append({
                "query": q["query"], "query_id": q["query_id"], "v": 2,
                "ir": v2[q["query"]]["ir"], "subs": v2[q["query"]]["subs"],
                "m3_kind": "rich_frozen", "append": False,
            })
        else:
            # Preserve + Augment：原始 v2 subqueries + M3.1 rescue subqueries（追加）
            lines.append({
                "query": q["query"], "query_id": q["query_id"], "v": 2,
                "ir": v2[q["query"]]["ir"],
                "subs": v2[q["query"]]["subs"] + rescue[q["query"]]["subs"],
                "m3_kind": "sparse_append", "append": True,
                "rescue_subqueries": [s["query_text"] for s in rescue[q["query"]]["subs"]],
                "m31_planner_version": rescue[q["query"]]["m31_planner_version"],
                "plan_hash": rescue[q["query"]]["plan_hash"],
            })
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    from collections import Counter
    cnt = Counter()
    for ln in lines:
        cnt.update(s["intent"] for s in ln["subs"])
    print(f"M3.1 plan cache：{len(lines)} 条（rich {len(rich)} + sparse_append {len(sparse)}）")
    print("类型统计：", dict(cnt))
    print("写", PLAN_FILE)


# --------------------------------------------------------------------------
# 合并 recall cache：pasa(87) + 被 M3 cache 覆盖的 M3.1 rescue 子查询（重放，0 联网）
# 返回 (merged, new_network) —— new_network = 真正需联网的新 rescue 子查询（无任何缓存）
# --------------------------------------------------------------------------
async def build_merged_recall_cache(plans: dict[str, dict]) -> tuple[dict[str, list], list[str]]:
    """合并 recall cache：pasa(87) + 被 M3 cache / M3.1 cache 覆盖的 rescue 子查询（重放，0 联网）。
    返回 (merged, new_network)。断点续跑：已在本 run 的 M3.1 cache（eval/cache/m31）里的子查询
    重放命中 → 不算 new_network，避免重复联网/虚高配额。"""
    merged = load_merged_cache() or {}
    # 确保 pasa 原始子查询齐全（persisted merged 可能不含全部 pasa）
    for k, v in load_recall_cache_map().items():
        merged.setdefault(k, v)
    cache_m3 = ResponseCache(M3_CACHE_DIR, mode="replay")
    cache_m31 = ResponseCache(CACHE_DIR, mode="replay")
    oa_m3 = OpenAlexAdapter(mailto=settings.openalex_mailto, cache=cache_m3)
    oa_m31 = OpenAlexAdapter(mailto=settings.openalex_mailto, cache=cache_m31)
    rescue_subs = {s["query_text"] for p in plans.values() if p.get("append")
                   for s in p["subs"] if s["query_text"] in p.get("rescue_subqueries", [])}
    new_network = []
    for q in sorted(rescue_subs):
        if q in merged:
            continue  # 已在 pasa / M3 / 本 run cache
        got = None
        for oa in (oa_m3, oa_m31):
            try:
                got = await oa.search(q, limit=20)  # 重放，0 联网
                break
            except CacheMiss:
                continue
        if got is not None:
            merged[q] = got
        else:
            new_network.append(q)  # 无任何缓存 → 本 run 将联网
    return merged, new_network


def write_merged_cache(merged: dict[str, list]) -> None:
    p = Path("eval/cache/m31/recall_cache.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for q, evs in merged.items():
            f.write(json.dumps({"q": q, "evs": [e.model_dump() for e in evs]}, ensure_ascii=False) + "\n")


def load_merged_cache() -> dict[str, list]:
    p = Path("eval/cache/m31/recall_cache.jsonl")
    if not p.exists():
        return {}
    return load_recall_cache_map(str(p))


# --------------------------------------------------------------------------
# 配额预检（联网前）：remaining < planned_new + margin 时拒绝；返回 planned_new
# --------------------------------------------------------------------------
def check_quota(planned_new: list[str]) -> int:
    n = len(set(planned_new))
    try:
        with urllib.request.urlopen(OPENALEX_PROBE, timeout=15) as r:
            hdr = dict((k.lower(), v) for k, v in r.headers.items())
        remaining = int(hdr.get("x-ratelimit-remaining", "0") or 0)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise SystemExit("!! OpenAlex 配额已耗尽（HTTP 429）。请切换 IP 恢复配额后重跑 search，中止不联网。")
        raise SystemExit(f"配额探测失败（HTTP {e.code}），中止，不执行联网检索。")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"配额探测失败（{type(e).__name__}: {e}），中止，不执行联网检索。")
    need = n + QUOTA_MARGIN
    print(f"配额预检：remaining={remaining}，planned_new_calls={n}，need≥{need}")
    if remaining < need:
        raise SystemExit(f"!! OpenAlex 配额不足（remaining {remaining} < {need}）。请切换 IP 恢复配额后重跑 search。当前不执行任何联网检索。")
    print("配额充足，可联网检索。")
    return n


def compute_budget(plans: dict[str, dict]) -> dict:
    """production_equivalent / new_network / cached_reuse（deterministic，不联网）。"""
    rescue_subs = {s["query_text"] for p in plans.values() if p.get("append")
                   for s in p["subs"] if s["query_text"] in p.get("rescue_subqueries", [])}
    orig_subs = {s["query_text"] for p in plans.values() if not p.get("append")
                 for s in p["subs"]} | \
                {s["query_text"] for p in plans.values() if p.get("append")
                 for s in p["subs"] if s["query_text"] not in p.get("rescue_subqueries", [])}
    v2_subs = {s["query_text"] for p in plans.values() for s in p["subs"] if not p.get("append")
               or s["query_text"] not in p.get("rescue_subqueries", [])}
    # 生产等价 logical = 全部唯一子查询（original v2 + m31 rescue）
    unique_all = {s["query_text"] for p in plans.values() for s in p["subs"]}
    return {
        "unique_orig_v2": len(orig_subs),
        "unique_rescue": len(rescue_subs),
        "production_equivalent": len(unique_all),
        "new_network": 0,  # 由 build_merged 填充
        "cached_reuse": 0,
    }


# --------------------------------------------------------------------------
# Phase C：search —— 22 条统一评测（cache-first；仅新 M3.1 rescue 子查询联网，预算化）
# --------------------------------------------------------------------------
async def cmd_search(limit: int | None, force: bool) -> None:
    plans = load_query_plans()
    if not plans:
        raise SystemExit("EXPERIMENT_INVALID: 无 M3.1 plan，先跑 build")
    queries = [q for q in select_queries() if q["query"] in plans]
    if limit:
        queries = queries[:limit]

    merged, new_network = await build_merged_recall_cache(plans)
    write_merged_cache(merged)  # 供离线 report 归因（pasa + M3 覆盖 + 新联网结果）
    budget = compute_budget(plans)
    budget["new_network"] = len(new_network)
    budget["cached_reuse"] = budget["unique_rescue"] - len(new_network)

    # Step 10：预算护栏 —— 超限即 STOP（不自动削减 query，不自动执行）
    print(f"=== 预算核算 ===")
    print(f"production_equivalent_logical_searches = {budget['production_equivalent']} "
          f"(orig {budget['unique_orig_v2']} + rescue {budget['unique_rescue']})  [预算 <= {PROD_EQUIV_BUDGET}]")
    print(f"new OpenAlex network = {budget['new_network']}  [预算 <= {NEW_NETWORK_BUDGET}]；"
          f"cached_reuse = {budget['cached_reuse']}")
    if budget["production_equivalent"] > PROD_EQUIV_BUDGET:
        raise SystemExit(
            f"!! BUDGET_EXCEEDED：production_equivalent {budget['production_equivalent']} > {PROD_EQUIV_BUDGET}。"
            f"STOP，不自动削减 query，不自动执行。")
    if budget["new_network"] > NEW_NETWORK_BUDGET:
        raise SystemExit(
            f"!! BUDGET_EXCEEDED：new OpenAlex network {budget['new_network']} > {NEW_NETWORK_BUDGET}。"
            f"STOP，不自动削减 query，不自动执行。")

    if not force:
        check_quota(new_network)

    print(f"M3.1 检索：{len(queries)} 条，assoc_safepass=True（KEEP_SAFEPASS），enable_citation_expansion=False")

    cache = ResponseCache(CACHE_DIR, mode="write")  # 仅新 M3.1 响应写入
    engine = SearchEngine(response_cache=cache, enable_citation_expansion=False, assoc_safepass=True)
    engine.load_plan_cache(str(PLAN_FILE))
    engine._recall_cache = merged  # 预加载：v2 + M3 cache 覆盖的 rescue 全命中，recall.search 不触发
    n_missing = sum(1 for p in plans.values() for s in p["subs"] if s["query_text"] not in merged)
    print(f"recall cache 预加载 {len(merged)} 条，uncovered={n_missing}（即联网的 new_network={budget['new_network']}）")

    M31_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    done_before = 0
    todo = []
    for q in queries:
        rep = M31_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            done_before += 1
            continue
        todo.append(q)
    if done_before:
        print(f"断点续跑：跳过已完成 {done_before} 条，剩余 {len(todo)} 条")

    for i, q in enumerate(todo, 1):
        qid = q["query_id"]
        gold_groups = match_gold(q)
        gold_titles = [g["title"] for g in q.get("gold", []) if g.get("title")]
        recorder = TraceRecorder(
            query_id=qid, run_id=EXPERIMENT, baseline_reference_id=BASELINE_REF, response_cache=cache)
        recorder.set_gold_titles(gold_titles)
        engine.recorder = recorder
        t0 = time.time()
        try:
            results, telemetry, traces = await engine.search_full(q["query"], top_k=TOP_K)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] {qid}: FAIL {type(e).__name__}: {e}")
            with open(M31_DIR / "search_failures.log", "a", encoding="utf-8") as f:
                f.write(f"{qid}\t{type(e).__name__}\t{e}\n")
            continue
        latency_ms = round((time.time() - t0) * 1000, 1)
        metrics = compute_p_r_f1(results, gold_groups)
        snapshots = {s["stage"]: s for s in recorder.candidate_snapshots}
        report = {
            "experiment": EXPERIMENT, "query_id": qid,
            "append": bool(plans[q["query"]].get("append")),
            "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
            "tp": metrics["tp"], "api_calls": telemetry.api_calls, "cache_hits": telemetry.cache_hits,
            "llm_calls": telemetry.llm_calls, "input_tokens": telemetry.input_tokens,
            "output_tokens": telemetry.output_tokens, "latency_ms": latency_ms,
            "n_predicted": len(results),
            "raw_candidates": snapshots.get("after_raw_recall", {}).get("candidate_count", 0),
            "rerank_pool_candidates": snapshots.get("rerank_pool", {}).get("candidate_count", 0),
        }
        recorder.save(TRACE_DIR)
        cache.flush()
        (M31_DIR / f"report_{qid}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        kind = "APPEND" if report["append"] else "rich"
        print(f"[{i}/{len(todo)}] {qid} [{kind}]: F1={report['f1']:.4f} P={report['precision']:.4f} "
              f"R={report['recall']:.4f} api={report['api_calls']} cache={report['cache_hits']} "
              f"llm={report['llm_calls']} raw={report['raw_candidates']} {latency_ms}ms")
    cache.close()

    # 联网护栏：new_network 应全部真正联网；api_calls 合计应为 new_network（logical）
    actual_api = sum(json.loads((M31_DIR / f"report_{q['query_id']}.json").read_text(encoding="utf-8"))
                     .get("api_calls", 0) for q in queries
                     if (M31_DIR / f"report_{q['query_id']}.json").exists())
    print(f"M3.1 联网护栏：实际 telemetry.api_calls 合计 = {actual_api}，预算 new_network = {budget['new_network']}。")
    if actual_api > budget["new_network"]:
        print(f"!! 注意：实际联网 > 预算 new_network（可能有重复子查询被多查）。")
    print("检索完成。生成产物...")
    cmd_report()


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def load_report(dirp: Path, qid: str) -> dict:
    return json.loads((dirp / f"report_{qid}.json").read_text(encoding="utf-8"))


def load_trace(dirp: Path, qid: str) -> dict:
    return json.loads((dirp / f"trace_{qid}.json").read_text(encoding="utf-8"))


def traces_raw_pool_final(dirp: Path, queries: list[dict]) -> dict:
    raw_i: set[str] = set(); raw_p: set[str] = set()
    pool_p: set[str] = set(); final_p: set[str] = set()
    for q in queries:
        trc = dirp / f"trace_{q['query_id']}.json"
        if not trc.exists():
            continue
        t = json.loads(trc.read_text(encoding="utf-8"))
        for g in t.get("gold_lifecycle", []):
            if g.get("first_seen_stage") in ("regular_recall", "assoc_recall"):
                raw_i.add(g["canonical_id"]); raw_p.add(g["title_n"])
            if g.get("survived_prekeep"):
                pool_p.add(g["title_n"])
            if g.get("final_rank") is not None:
                final_p.add(g["title_n"])
    return {
        "raw_instances": len(raw_i), "raw_papers": len(raw_p),
        "pool_papers": len(pool_p), "final_papers": len(final_p),
        "raw_to_pool": (len(pool_p) / len(raw_p)) if raw_p else 0.0,
        "pool_to_final": (len(final_p) / len(pool_p)) if pool_p else 0.0,
    }


def sparse_gold_stats(dirp: Path, queries: list[dict], plans: dict[str, dict],
                      rescue_plans: dict[str, dict], merged: dict[str, list]) -> dict:
    """15 条 sparse 的 preserved / incremental / lost 归因（论文级 title_n，recall-cache 归属）。"""
    merged_titles = {sq: {norm_title(e.identity.title) for e in evs
                          if e.identity and e.identity.title} for sq, evs in merged.items()}
    preserved: set[str] = set()
    incremental: set[str] = set()
    lost_prepool: set[str] = set()
    lost_rerank: set[str] = set()
    # per-type incremental：intent -> set[title_n]
    type_incremental: dict[str, set[str]] = {t: set() for t in RESCUE_INTENTS}
    for q in queries:
        if q["query"] not in rescue_plans:
            continue
        rescue_subs = {s["query_text"] for s in rescue_plans[q["query"]]["subs"]}
        original_subs = {s["query_text"] for s in plans[q["query"]]["subs"]} - rescue_subs
        orig_titles = {t for sq in original_subs for t in merged_titles.get(sq, set())}
        trc = dirp / f"trace_{q['query_id']}.json"
        if not trc.exists():
            continue
        t = json.loads(trc.read_text(encoding="utf-8"))
        for g in t.get("gold_lifecycle", []):
            if g.get("first_seen_stage") not in ("regular_recall", "assoc_recall"):
                continue
            tn = g["title_n"]
            if tn in orig_titles:
                preserved.add(tn)
                continue
            # incremental：rescue 子查询 first-seen，且不在原始子查询结果
            fq = g.get("first_seen_query", "")
            intent = None
            for s in rescue_plans[q["query"]]["subs"]:
                if fq.startswith(s["query_text"][:40]) or s["query_text"].startswith(fq[:40]):
                    intent = s["intent"]; break
            if intent is None:
                continue
            incremental.add(tn)
            type_incremental[intent].add(tn)
            if not g.get("survived_prekeep"):
                lost_prepool.add(tn)
            elif g.get("final_rank") is None:
                lost_rerank.add(tn)
    return {
        "preserved_gold": len(preserved),
        "incremental_rescue_gold": len(incremental),
        "rescue_gold_lost_prepool": len(lost_prepool),
        "rescue_gold_lost_rerank": len(lost_rerank),
        "type_incremental": {t: len(v) for t, v in type_incremental.items()},
    }


def cmd_report() -> None:
    queries = select_queries()
    v2 = load_v2_plans(V2_PLAN_CACHE)
    rescue_plans = load_rescue_plans()
    sparse, rich = classify_sparse_rich(v2)
    b0 = load_nocit_baseline()
    meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}

    plans = load_query_plans()
    merged = load_merged_cache()  # 离线：读取 search 时持久化的合并 recall cache

    reports: dict[str, dict] = {}
    for q in queries:
        rep = M31_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            reports[q["query_id"]] = json.loads(rep.read_text(encoding="utf-8"))
    n = len(reports)

    # ---- 全 gold 生命周期 ----
    all_lifecycle: list[dict] = []
    rescue_texts = {q["query_id"]: {s["query_text"][:80]: s["intent"] for s in rescue_plans[q["query"]]["subs"]}
                    for q in queries if q["query"] in rescue_plans}
    for q in queries:
        qid = q["query_id"]
        if qid not in reports:
            continue
        trc = load_trace(TRACE_DIR, qid)
        for g in trc.get("gold_lifecycle", []):
            fq = g.get("first_seen_query", "")[:80]
            intent = ""; src = "rich"
            if qid in rescue_texts and fq in rescue_texts[qid]:
                intent = rescue_texts[qid][fq]; src = "rescue"
            row = {**g, "query_id": qid, "query_type": intent, "source_kind": src,
                   "append": bool(plans.get(q["query"], {}).get("append"))}
            all_lifecycle.append(row)
    lc_cols = ["query_id", "canonical_id", "title_n", "first_seen_stage", "first_seen_query",
               "first_seen_rank", "query_type", "source_kind", "append", "survived_prekeep",
               "pre_rerank_rank", "reranker_rank", "final_rank", "drop_stage", "drop_reason"]
    with open(M31_DIR / "m31_gold_lifecycle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=lc_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_lifecycle:
            w.writerow(row)

    # ---- per-type metrics（generated=raw_counts / executed=final subs / incremental / pruned）----
    type_agg = {t: {"generated": 0, "executed": 0, "incremental": 0} for t in RESCUE_INTENTS}
    sparse_metrics: list[dict] = []
    stats = sparse_gold_stats(TRACE_DIR, queries, plans, rescue_plans, merged)
    for q in queries:
        qid = q["query_id"]
        if q["query"] not in rescue_plans or qid not in reports:
            continue
        rp = rescue_plans[q["query"]]
        raw = rp.get("raw_counts", {})
        executed = {t: sum(1 for s in rp["subs"] if s["intent"] == t) for t in RESCUE_INTENTS}
        row = {"query_id": qid}
        for t in RESCUE_INTENTS:
            row[f"{t}_generated"] = raw.get(t, 0)
            row[f"{t}_executed"] = executed[t]
            row[f"{t}_pruned"] = max(0, raw.get(t, 0) - executed[t])
            type_agg[t]["generated"] += raw.get(t, 0)
            type_agg[t]["executed"] += executed[t]
        # incremental per type（论文级）——从 sparse_gold_stats 的 type_incremental 按 query 归属近似：
        # 这里用全量 type_incremental 总量，逐 query 明细依赖 trace（此处给出总量）。
        sparse_metrics.append(row)
    with open(M31_DIR / "m31_query_type_metrics.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id"] + [f"{t}_{c}" for t in RESCUE_INTENTS for c in
                               ("generated", "executed", "pruned", "incremental", "inc_per_exec")] + ["append"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        # 逐 query incremental（用 trace first_seen 按 query+type 归因）
        q_type_inc: dict[tuple[str, str], set[str]] = {}
        for q in queries:
            if q["query"] not in rescue_plans or q["query_id"] not in reports:
                continue
            rp = rescue_plans[q["query"]]
            original_subs = {s["query_text"] for s in plans[q["query"]]["subs"]} - \
                            {s["query_text"] for s in rp["subs"]}
            orig_titles = {t for sq in original_subs for t in
                           {norm_title(e.identity.title) for e in merged.get(sq, [])
                            if e.identity and e.identity.title}}
            trc = load_trace(TRACE_DIR, q["query_id"])
            for g in trc.get("gold_lifecycle", []):
                if g.get("first_seen_stage") not in ("regular_recall", "assoc_recall"):
                    continue
                tn = g["title_n"]
                if tn in orig_titles:
                    continue
                fq = g.get("first_seen_query", "")
                for s in rp["subs"]:
                    if fq.startswith(s["query_text"][:40]) or s["query_text"].startswith(fq[:40]):
                        q_type_inc.setdefault((q["query_id"], s["intent"]), set()).add(tn)
                        break
        for row in sparse_metrics:
            qid = row["query_id"]
            for t in RESCUE_INTENTS:
                inc = len(q_type_inc.get((qid, t), set()))
                row[f"{t}_incremental"] = inc
                row[f"{t}_inc_per_exec"] = round(inc / row[f"{t}_executed"], 3) if row[f"{t}_executed"] else 0.0
                type_agg[t]["incremental"] += inc
            row["append"] = True
            w.writerow(row)
        w.writerow({})
        w.writerow({"query_id": "TOTAL", **{f"{t}_generated": type_agg[t]["generated"] for t in RESCUE_INTENTS},
                    **{f"{t}_executed": type_agg[t]["executed"] for t in RESCUE_INTENTS},
                    **{f"{t}_pruned": type_agg[t]["generated"] - type_agg[t]["executed"] for t in RESCUE_INTENTS},
                    **{f"{t}_incremental": type_agg[t]["incremental"] for t in RESCUE_INTENTS}})

    # ---- 三方漏斗 ----
    m31_funnel = traces_raw_pool_final(TRACE_DIR, queries)
    m3r_funnel = traces_raw_pool_final(M3R_TRACE_DIR, queries)
    b0_funnel = traces_raw_pool_final(Path("eval/diagnostics/v2_instrumented"), queries)

    # ---- 三方逐 query 对比 ----
    total_phys = {k: 0 for k in ("b0", "m3r", "m31")}
    total_api = {k: 0 for k in ("b0", "m3r", "m31")}
    total_llm = {k: 0 for k in ("b0", "m3r", "m31")}
    rows = []
    for qid in sorted(reports):
        r = reports[qid]
        b = b0[qid]
        rrep = load_report(M3R_RUN_DIR, qid) if (M3R_RUN_DIR / f"report_{qid}.json").exists() else {}
        total_phys["b0"] += b.get("api_calls", 0)
        total_api["b0"] += b.get("api_calls", 0)
        total_phys["m3r"] += rrep.get("api_calls", 0)
        total_api["m3r"] += rrep.get("api_calls", 0)
        total_phys["m31"] += r.get("api_calls", 0)
        total_api["m31"] += r.get("api_calls", 0)
        total_llm["b0"] += b.get("llm_calls", 0)
        total_llm["m3r"] += rrep.get("llm_calls", 0)
        total_llm["m31"] += r.get("llm_calls", 0)
        rows.append({
            "query_id": qid, "kind": "APPEND" if r.get("append") else "rich",
            "b0_f1": b["f1"], "b0_prec": b["precision"], "b0_recall": b["recall"],
            "m3r_f1": rrep.get("f1", ""), "m3r_prec": rrep.get("precision", ""), "m3r_recall": rrep.get("recall", ""),
            "m31_f1": r["f1"], "m31_prec": r["precision"], "m31_recall": r["recall"],
            "m31_raw_gold": sum(1 for row2 in all_lifecycle if row2["query_id"] == qid and
                                row2["first_seen_stage"] in ("regular_recall", "assoc_recall")),
            "m31_final_gold": sum(1 for row2 in all_lifecycle if row2["query_id"] == qid and row2.get("final_rank") is not None),
        })
    with open(M31_DIR / "m31_vs_b0_vs_m3r.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "kind", "b0_f1", "b0_prec", "b0_recall", "m3r_f1", "m3r_prec", "m3r_recall",
                "m31_f1", "m31_prec", "m31_recall", "m31_raw_gold", "m31_final_gold"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        w.writerow({})
        def mean(vals): return round(sum(vals) / len(vals), 4) if vals else ""
        m3r_f1s = [float(r["m3r_f1"]) for r in rows if r["m3r_f1"] != ""]
        m3r_precs = [float(r["m3r_prec"]) for r in rows if r["m3r_prec"] != ""]
        m3r_recs = [float(r["m3r_recall"]) for r in rows if r["m3r_recall"] != ""]
        w.writerow({
            "query_id": "MEAN", "kind": "",
            "b0_f1": mean([r["b0_f1"] for r in rows]), "b0_prec": mean([r["b0_prec"] for r in rows]),
            "b0_recall": mean([r["b0_recall"] for r in rows]),
            "m3r_f1": mean(m3r_f1s), "m3r_prec": mean(m3r_precs), "m3r_recall": mean(m3r_recs),
            "m31_f1": mean([r["m31_f1"] for r in rows]), "m31_prec": mean([r["m31_prec"] for r in rows]),
            "m31_recall": mean([r["m31_recall"] for r in rows]),
            "m31_raw_gold": m31_funnel["raw_papers"], "m31_final_gold": m31_funnel["final_papers"],
        })

    # ---- 成本分解（三方：生产等价 logical / 物理 HTTP / 复用）----
    budget = compute_budget(plans) if plans else {}
    budget["new_network"] = sum(r.get("api_calls", 0) for r in reports.values()) if reports else 0
    budget["cached_reuse"] = budget["unique_rescue"] - budget["new_network"]
    cost_rows = [
        {"variant": "B0", "production_equiv_logical": 87, "physical_http_new": 87, "cached_reused": 0,
         "reranker_llm_calls": total_llm["b0"]},
        {"variant": "M3-R_APPEND", "production_equiv_logical": 141, "physical_http_new": 0, "cached_reused": 52,
         "reranker_llm_calls": total_llm["m3r"]},
        {"variant": "M3.1", "production_equiv_logical": budget.get("production_equivalent", ""),
         "physical_http_new": budget["new_network"], "cached_reused": budget.get("cached_reuse", ""),
         "reranker_llm_calls": total_llm["m31"]},
    ]
    with open(M31_DIR / "m31_cost_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "production_equiv_logical", "physical_http_new",
                                          "cached_reused", "reranker_llm_calls"])
        w.writeheader()
        for row in cost_rows:
            w.writerow(row)

    # ---- 决策 ----
    raw = m31_funnel["raw_papers"]
    inc = stats["incremental_rescue_gold"]
    inc_total = inc
    inc_lost = stats["rescue_gold_lost_prepool"] + stats["rescue_gold_lost_rerank"]
    inc_retained = inc_total - inc_lost
    lines = []
    lines.append("# M3.1_PLANNER_PROMPT_REVISION 决策报告")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}（M3.1 全量 {n}/{len(queries)}-query 统一评测）")
    lines.append(f"- M3_1_PLANNER_VERSION={M3_1_PLANNER_VERSION}，prompt_hash={meta.get('prompt_hash', '?')[:16]}…，"
                 f"plan_hash={meta.get('plan_hash', '?')[:16]}…")
    lines.append(f"- 对照：B0=`{BASELINE_REF}`（F1=0.0618 R=0.1339 raw=22）；M3-R_APPEND=`{M3R_REF}`（F1=0.0664 R=0.1492 raw=25）。")
    lines.append("- 本轮唯一变量 vs M3-R：SparsePlanRescue Prompt（M3.1）。15 条 sparse = original_v2_subqueries + "
                 "M3.1_rescue_subqueries（Preserve+Augment）；7 条 rich 原样。")
    lines.append("- 冻结：assoc_safepass=True、Citation/Reference/Metadata OFF、OpenAlex 不变、top_k=20、Prekeep/Reranker 不变。")
    lines.append("")
    lines.append("## 1. 三方保留漏斗（论文级 unique Gold papers，title_n；B0= safepass ON）")
    lines.append("")
    lines.append("| 阶段 | B0 | M3-R_APPEND | M3.1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| raw unique Gold papers | {b0_funnel['raw_papers']} | {m3r_funnel['raw_papers']} | **{m31_funnel['raw_papers']}** |")
    lines.append(f"| pool unique Gold papers | {b0_funnel['pool_papers']} | {m3r_funnel['pool_papers']} | {m31_funnel['pool_papers']} |")
    lines.append(f"| final unique Gold papers | {b0_funnel['final_papers']} | {m3r_funnel['final_papers']} | {m31_funnel['final_papers']} |")
    lines.append(f"| raw query-gold instances | {b0_funnel['raw_instances']} | {m3r_funnel['raw_instances']} | {m31_funnel['raw_instances']} |")
    lines.append(f"| raw→pool | {b0_funnel['raw_to_pool']:.1%} | {m3r_funnel['raw_to_pool']:.1%} | {m31_funnel['raw_to_pool']:.1%} |")
    lines.append(f"| pool→final | {b0_funnel['pool_to_final']:.1%} | {m3r_funnel['pool_to_final']:.1%} | {m31_funnel['pool_to_final']:.1%} |")
    lines.append("")
    lines.append("## 2. 核心指标（三方）")
    lines.append("")
    lines.append("| 指标 | B0 | M3-R_APPEND | M3.1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| mean F1 | {sum(r['b0_f1'] for r in rows)/n:.4f} | {mean(m3r_f1s)} | {sum(r['m31_f1'] for r in rows)/n:.4f} |")
    lines.append(f"| mean Precision | {sum(r['b0_prec'] for r in rows)/n:.4f} | {mean(m3r_precs)} | {sum(r['m31_prec'] for r in rows)/n:.4f} |")
    lines.append(f"| mean Recall | {sum(r['b0_recall'] for r in rows)/n:.4f} | {mean(m3r_recs)} | {sum(r['m31_recall'] for r in rows)/n:.4f} |")
    lines.append(f"| OpenAlex logical calls | {total_api['b0']} | {total_api['m3r']} | {total_api['m31']} |")
    lines.append(f"| OpenAlex physical HTTP（新联网） | {total_phys['b0']} | {total_phys['m3r']} | {total_phys['m31']} |")
    lines.append(f"| Reranker(LLM) calls | {total_llm['b0']} | {total_llm['m3r']} | {total_llm['m31']} |")
    lines.append("")
    lines.append("## 3. Per-type CORE/ANCHOR/DISCOVERY（15 条 sparse；论文级 title_n）")
    lines.append("")
    lines.append("| query_type | generated | executed | pruned | incremental Gold | Gold/executed |")
    lines.append("|---|---|---|---|---|---|")
    for t in RESCUE_INTENTS:
        gen = type_agg[t]["generated"]; exe = type_agg[t]["executed"]
        inc_t = stats["type_incremental"][t]
        lines.append(f"| {t} | {gen} | {exe} | {gen - exe} | {inc_t} | "
                     f"{round(inc_t / exe, 3) if exe else 0.0} |")
    lines.append("")
    lines.append("## 4. Sparse 归因（论文级 title_n）")
    lines.append("")
    lines.append(f"- preserved Gold（原始 sparse 子查询保留）= **{stats['preserved_gold']}**")
    lines.append(f"- incremental Rescue Gold（M3.1 rescue 子查询新增）= **{inc}**")
    lines.append(f"- Rescue Gold lost before pool = **{stats['rescue_gold_lost_prepool']}**；"
                 f"lost by reranker = **{stats['rescue_gold_lost_rerank']}**")
    if inc_total > 0:
        lines.append(f"- 新增 Rescue Gold raw→final 保留率 = {inc_retained/inc_total:.0%}（{inc_retained}/{inc_total}）")
    lines.append("")
    lines.append("## 5. 决策 Gate（M3.1 raw_unique_gold，基线 B0=22 / M3-R=25；Step 12）")
    lines.append("")
    if raw >= 35:
        gate = "STRONG_QUERY_FORMULATION_SUCCESS"
        gate_note = "raw≥35。本轮完成后 STOP Query Formulation tuning；若 final F1 未同步提升 → NEXT=M4_RERANKER_RETENTION。"
    elif raw >= 30:
        gate = "QUERY_FORMULATION_VALIDATED"
        gate_note = "raw 30~34。本轮完成后 STOP Query Formulation tuning。"
    elif raw >= 26:
        gate = "QUERY_FORMULATION_TUNING_STOP"
        gate_note = "raw 26~29：保留本版（相对 M3-R 有提升），但停止 Query Formulation Prompt tuning。"
    else:
        gate = "QUERY_FORMULATION_TUNING_STOP"
        gate_note = "raw≤25：回退到 M3-R/B0 中更优者，停止 Query Formulation Prompt tuning。"
    lines.append(f"- **M3.1 raw_unique_gold_papers = {raw}**（B0 {b0_funnel['raw_papers']} / M3-R {m3r_funnel['raw_papers']}）。")
    lines.append(f"- **判定：{gate}**。{gate_note}")
    lines.append("")
    lines.append("## 6. F1 判定（Step 13）")
    lines.append("")
    m31_f1 = sum(r["m31_f1"] for r in rows) / n
    m3r_f1 = sum(m3r_f1s) / len(m3r_f1s) if m3r_f1s else 0.0
    b0_f1 = sum(r["b0_f1"] for r in rows) / n
    if raw > m3r_funnel["raw_papers"] and m31_f1 <= max(b0_f1, m3r_f1):
        lines.append(f"raw 提升（{m3r_funnel['raw_papers']}→{raw}）但 final F1 未同步提升（B0 {b0_f1:.4f} / M3-R {m3r_f1:.4f} / M3.1 {m31_f1:.4f}）。")
        lines.append("**NEXT = M4_RERANKER_RETENTION**（瓶颈在 final 保留，非 query formulation）。")
    elif raw <= m3r_funnel["raw_papers"]:
        lines.append(f"raw 未超过 M3-R（{raw} ≤ {m3r_funnel['raw_papers']}）。")
        lines.append("**如实记录 Query Formulation 天花板，不把瓶颈归因给 reranker。** 保持已批准的最佳版本；本轮 STOP。")
    else:
        lines.append(f"raw 与 final F1 同向（或 F1 同步提升）。")
    lines.append("")
    lines.append("## 7. 成本（Step 0/10/11；deterministic）")
    lines.append("")
    lines.append(f"- **production_equivalent_logical_searches**：B0=87，M3-R=141，M3.1={budget.get('production_equivalent','?')}（orig {budget.get('unique_orig_v2','?')} + rescue {budget.get('unique_rescue','?')}）")
    lines.append(f"- **replay / cached_reused**：M3-R=52；M3.1={budget.get('cached_reuse','?')}（命中 pasa / M3 缓存，0 联网）")
    lines.append(f"- **physical HTTP new**：M3-R=0；M3.1={budget['new_network']}（<= {NEW_NETWORK_BUDGET}）")
    lines.append("")
    lines.append("## 8. Gold isolation 校验（Step 9）")
    lines.append("")
    lines.append("- Production Planner 输入仅 question 文本；未接触 gold title/author/DOI/arXiv/abstract/Oracle probe/per-Gold taxonomy。")
    lines.append("- 若发现泄漏 → EXPERIMENT_INVALID=true。本轮检查：Prompts 含 Gold isolation 规则；raw_counts 仅记录 LLM 输出结构，无 gold 信息。")
    lines.append("- Gold 仅在 evaluator（TraceRecorder）于 retrieval 后读取。")
    lines.append("")
    lines.append("**本轮（M3.1）到此为止：无论结果如何，停止 Query Formulation Prompt tuning；完成后 STOP，不自动进入 M4。**")
    (M31_DIR / "m31_decision.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== M3.1 汇总 ===")
    print(f"B0 raw={b0_funnel['raw_papers']}  M3-R raw={m3r_funnel['raw_papers']}  M3.1 raw={raw}")
    print(f"M3.1 pool={m31_funnel['pool_papers']}  final={m31_funnel['final_papers']}  instances={m31_funnel['raw_instances']}")
    print(f"F1  B0={b0_f1:.4f}  M3-R={m3r_f1:.4f}  M3.1={m31_f1:.4f}")
    print(f"preserved={stats['preserved_gold']}  incremental_rescue={inc}  lost_prepool={stats['rescue_gold_lost_prepool']}  lost_rerank={stats['rescue_gold_lost_rerank']}")
    print(f"per-type inc: " + ", ".join(f"{t}={stats['type_incremental'][t]}" for t in RESCUE_INTENTS))
    print(f"cost: production_equiv={budget.get('production_equivalent','?')}  cached_reuse={budget.get('cached_reuse','?')}  physical_http_new={budget['new_network']}")
    print(f"Gate：{gate}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "build", "search", "report", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="跳过配额预检强制联网（不推荐）")
    args = ap.parse_args()
    if args.mode in ("generate", "all"):
        await cmd_generate(args.limit)
    if args.mode in ("build", "all"):
        cmd_build()
    if args.mode in ("search", "all"):
        await cmd_search(args.limit, force=args.force)
    if args.mode == "report":
        cmd_report()


if __name__ == "__main__":
    asyncio.run(main())
