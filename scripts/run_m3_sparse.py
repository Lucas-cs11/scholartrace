"""M3_SPARSE_PLAN_RESCUE：稀疏查询救援（只解决 15 条 sparse plan 的 Query Formulation）。

对照：B0 = PASA_ASSOC_NO_CIT（F1=0.0618, P=0.0615, R=0.1339, api=87）——不重跑，读基线报告。
触发：assoc_count==0 或 total_subquery_count<=1（=15 条只有 1 条 thin 子查询的 query）。
7 条 rich plan 冻结原样（含 v2 assoc 联想词锚点，是 raw=22 的来源），**不重新调用**。

M3 唯一改动：15 条 sparse query 用 SparsePlanRescue 重写 plan（≤4 条：core×1+anchor×1-2+
discovery×0-1）。Retriever/Prekeep/Reranker/OpenAlex/top_k=20 全部冻结；assoc_safepass=True
（M2 结论 KEEP_SAFEPASS）；Citation/Reference/Metadata 扩展关闭。

缓存/网络策略：
- 现有 v2/rich-plan 检索全部读缓存（_pasa_recall_cache.jsonl 预加载 → 0 联网）。
- 仅新的 M3 Rescue 子查询（不在 recall cache 的文本）请求 OpenAlex。
- 不重新在线执行原始 89 条 v2 子查询。最大 15×4=60 次新检索。

Gold isolation（EXPERIMENT_INVALID 门禁）：rescue planner 只喂 question 文本；gold 只进
TraceRecorder 诊断。严禁基于 Gold 生成论文标题/作者/DOI/arXiv ID。

产物（eval/runs/m3_sparse_plan_rescue/）：
  m3_sparse_plans.jsonl      —— 15 条 rescue plan（含 m3_planner_version / per-query plan_hash）
  m3_plan_meta.json          —— M3_PLANNER_VERSION + 全量 plan_hash
  m3_query_plans.jsonl       —— 22 条统一 plan cache（7 rich 冻结原样 + 15 rescue；v=2，供引擎加载）
  m3_sparse_query_metrics.csv —— 15 条 sparse query：rescue前/后 raw Gold、CORE/ANCHOR/DISCOVERY 各 new Gold
  m3_gold_lifecycle.csv      —— 全 gold 生命周期（含 m3_new + RETRIEVAL_GAIN vs RERANK_LOSS）
  m3_vs_b0.csv               —— 逐 query B0 vs M3 + MEAN 聚合
  m3_decision.md             —— 决策报告（paper-level raw gate）

用法：
  python scripts/run_m3_sparse.py generate          # 生成 15 条 rescue plan（LLM，不联网 OpenAlex）
  python scripts/run_m3_sparse.py build             # 组合 rich(冻结)+rescue -> m3 plan cache（离线）
  python scripts/run_m3_sparse.py search            # 22 条统一评测（rich 读缓存；rescue 联网，先验配额）
  python scripts/run_m3_sparse.py report            # 只重生成产物/决策（不联网）
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

from eval.harness import compute_p_r_f1, match_gold
from src.observability.canonical import is_gold_by_title, norm_title
from src.observability.response_cache import ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.planner import ASSOC_INTENT
from src.planner_rescue import (
    INTENT_ANCHOR, INTENT_CORE, INTENT_DISCOVERY, M3_PLANNER_VERSION, SparsePlanRescue,
)
from src.schemas import PaperEvidence
from src.search import SearchEngine
from scripts.eval_benchmark import load_pasa

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
V2_PLAN_CACHE = "eval/runs/_pasa_plan_cache.jsonl"
NOCIT_RESULTS = "eval/runs/nocitation/nocitation_results.json"
RECALL_CACHE = "eval/runs/_pasa_recall_cache.jsonl"
M3_DIR = Path("eval/runs/m3_sparse_plan_rescue")
RESCUE_PLAN_FILE = M3_DIR / "m3_sparse_plans.jsonl"
META_FILE = M3_DIR / "m3_plan_meta.json"
PLAN_FILE = M3_DIR / "m3_query_plans.jsonl"
CACHE_DIR = "eval/cache/m3_sparse_plan_rescue"
TRACE_DIR = Path("eval/diagnostics/m3_sparse_plan_rescue")
EXPERIMENT = "PASA_ASSOC_M3_SPARSE_RESCUE"
BASELINE_REF = "PASA_ASSOC_NO_CIT"
TOP_K = 20
RESCUE_INTENTS = (INTENT_CORE, INTENT_ANCHOR, INTENT_DISCOVERY)
QUOTA_MARGIN = 20  # 联网前保留余量：remaining < planned_new + margin 时拒绝运行
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


def load_m3_plans() -> dict[str, dict]:
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


def load_recall_cache_map() -> dict[str, list]:
    m: dict[str, list] = {}
    if not Path(RECALL_CACHE).exists():
        return m
    for line in Path(RECALL_CACHE).read_text(encoding="utf-8").splitlines():
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
    """触发条件：assoc_count==0 或 total_subquery_count<=1 → sparse；否则 rich。"""
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


def plan_hash_of(plans: list[dict]) -> str:
    """M3 全量 plan hash：版本 + 所有 rescue plan 的规范序列。7 rich 冻结不参与（未变更）。"""
    canon = {"m3_planner_version": M3_PLANNER_VERSION,
             "plans": [{**{k: p[k] for k in ("query", "subs")}} for p in sorted(plans, key=lambda x: x["query"])]}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --------------------------------------------------------------------------
# Phase A：generate —— 15 条 sparse 的 rescue plan（LLM，一次 / query）
# --------------------------------------------------------------------------
async def cmd_generate(limit: int | None) -> None:
    M3_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_v2_plans(V2_PLAN_CACHE)
    sparse, rich = classify_sparse_rich(v2)
    existing = load_rescue_plans()
    todo = [q for q in sorted(sparse, key=lambda x: query_id_map()[x]) if q not in existing]
    if limit:
        todo = todo[:limit]
    print(f"M3 rescue plan 生成：sparse {len(sparse)} 条（rich {len(rich)} 条冻结不动），"
          f"已存在 {len(existing)} 条，本次生成 {len(todo)} 条")
    planner = SparsePlanRescue()
    for i, q in enumerate(todo, 1):
        t0 = time.time()
        subs = await planner.plan_raw(q)
        plan = {
            "query": q,
            "query_id": query_id_map()[q],
            "m3_planner_version": M3_PLANNER_VERSION,
            "subs": [s.model_dump() for s in subs],
        }
        plan["plan_hash"] = hashlib.sha256(
            json.dumps({"v": M3_PLANNER_VERSION, "subs": plan["subs"]}, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with open(RESCUE_PLAN_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(plan, ensure_ascii=False) + "\n")
        tag = " ".join(f"{s['intent']}={s['query_text'][:28]}" for s in plan["subs"])
        print(f"[{i}/{len(todo)}] {plan['query_id']} ({round((time.time() - t0) * 1000)}ms) {tag}")
    all_rescue = sorted(load_rescue_plans().values(), key=lambda x: x["query"])
    META_FILE.write_text(json.dumps({
        "m3_planner_version": M3_PLANNER_VERSION,
        "plan_hash": plan_hash_of(all_rescue),
        "sparse_count": len(sparse), "rich_count": len(rich),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    from collections import Counter
    cnt = Counter()
    for p in all_rescue:
        cnt.update(s["intent"] for s in p["subs"])
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    print("M3 rescue plan 类型统计：", dict(cnt))
    print(f"M3_PLANNER_VERSION={M3_PLANNER_VERSION}  plan_hash={meta['plan_hash'][:16]}…")
    print("写", RESCUE_PLAN_FILE)


# --------------------------------------------------------------------------
# Phase B：build —— rich(冻结原样) + rescue -> 22 条统一 plan cache（离线）
# --------------------------------------------------------------------------
def cmd_build() -> None:
    M3_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_v2_plans(V2_PLAN_CACHE)
    rescue = load_rescue_plans()
    sparse, rich = classify_sparse_rich(v2)
    missing = sparse - set(rescue)
    if missing:
        raise SystemExit(f"EXPERIMENT_INVALID: 缺 {len(missing)} 条 rescue plan，先跑 generate：{sorted(missing)[:3]}…")
    if rich != (set(v2) - sparse):
        raise SystemExit("EXPERIMENT_INVALID: sparse/rich 划分不一致")
    queries = select_queries()
    lines = []
    for q in queries:
        if q["query"] in rich:
            lines.append({
                "query": q["query"], "query_id": q["query_id"], "v": 2,
                "ir": v2[q["query"]]["ir"], "subs": v2[q["query"]]["subs"],
                "m3_kind": "rich_frozen", "rescue_query": False,
            })
        else:
            lines.append({
                "query": q["query"], "query_id": q["query_id"], "v": 2,
                "ir": v2[q["query"]]["ir"],  # 复用 v2 冻结 ir：reranker 看到的 query+ir 与 B0 一致
                "subs": rescue[q["query"]]["subs"],
                "m3_kind": "sparse_rescue", "rescue_query": True,
                "m3_planner_version": rescue[q["query"]]["m3_planner_version"],
                "plan_hash": rescue[q["query"]]["plan_hash"],
            })
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    from collections import Counter
    cnt = Counter()
    for ln in lines:
        cnt.update(s["intent"] for s in ln["subs"])
    print(f"M3 plan cache：{len(lines)} 条（rich {len(rich)} + rescue {len(sparse)}）")
    print("类型统计：", dict(cnt))
    print("写", PLAN_FILE)


# --------------------------------------------------------------------------
# 配额预检（联网前）：remaining < planned_new + margin 时拒绝，避免烧配额
# --------------------------------------------------------------------------
def check_quota(plans: dict[str, dict]) -> int:
    """探测 OpenAlex 配额。配额不足/探测失败时 raise SystemExit（不执行任何检索）。
    返回 planned_new_calls（将联网的新 rescue 子查询数）。"""
    recall = load_recall_cache_map()
    planned = [s["query_text"] for p in plans.values() if p.get("rescue_query")
               for s in p["subs"] if s["query_text"] not in recall]
    planned_n = len(set(planned))
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
    need = planned_n + QUOTA_MARGIN
    print(f"配额预检：remaining={remaining}，planned_new_calls={planned_n}，need≥{need}")
    if remaining < need:
        raise SystemExit(
            f"!! OpenAlex 配额不足（remaining {remaining} < {need}）。请切换 IP 恢复配额后重跑 search。"
            f"当前不执行任何联网检索。")
    print("配额充足，可联网检索。")
    return planned_n


# --------------------------------------------------------------------------
# Phase C：search —— 22 条统一评测（rich 读缓存 0 联网；rescue 新子查询联网）
# --------------------------------------------------------------------------
async def cmd_search(limit: int | None, force: bool) -> None:
    plans = load_m3_plans()
    if not plans:
        raise SystemExit("EXPERIMENT_INVALID: 无 M3 plan，先跑 build")
    queries = [q for q in select_queries() if q["query"] in plans]
    if limit:
        queries = queries[:limit]

    if not force:
        check_quota(plans)  # 不足时 raise SystemExit，绝不联网
    print(f"M3 检索：{len(queries)} 条，assoc_safepass=True（KEEP_SAFEPASS），enable_citation_expansion=False")

    cache = ResponseCache(CACHE_DIR, mode="write")
    engine = SearchEngine(response_cache=cache, enable_citation_expansion=False, assoc_safepass=True)
    engine.load_plan_cache(str(PLAN_FILE))
    engine.load_recall_cache(RECALL_CACHE)  # 预加载 v2 recall：rich 子查询全命中，0 联网
    n_loaded = len(engine._recall_cache)
    # 统计：哪些 rescue 子查询不在缓存（将联网），rich 子查询应全部命中
    missing = sorted({s["query_text"] for p in plans.values() if p.get("rescue_query")
                      for s in p["subs"] if s["query_text"] not in engine._recall_cache})
    rich_missing = sorted({s["query_text"] for p in plans.values() if not p.get("rescue_query")
                           for s in p["subs"] if s["query_text"] not in engine._recall_cache})
    print(f"recall cache 预加载 {n_loaded} 条。rescue 新子查询将联网 {len(missing)} 条；"
          f"rich 子查询缺失（不应发生，应全缓存）{len(rich_missing)} 条")
    if rich_missing:
        print("!! 注意：rich 子查询未命中缓存，将联网（违背『不重跑 v2』），请检查 recall cache。")

    M3_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    done_before = 0
    todo = []
    for q in queries:
        rep = M3_DIR / f"report_{q['query_id']}.json"
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
            query_id=qid, run_id=EXPERIMENT, baseline_reference_id=BASELINE_REF,
            response_cache=cache,
        )
        recorder.set_gold_titles(gold_titles)
        engine.recorder = recorder

        t0 = time.time()
        try:
            results, telemetry, traces = await engine.search_full(q["query"], top_k=TOP_K)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(todo)}] {qid}: FAIL {type(e).__name__}: {e}")
            with open(M3_DIR / "search_failures.log", "a", encoding="utf-8") as f:
                f.write(f"{qid}\t{type(e).__name__}\t{e}\n")
            continue
        latency_ms = round((time.time() - t0) * 1000, 1)

        metrics = compute_p_r_f1(results, gold_groups)
        snapshots = {s["stage"]: s for s in recorder.candidate_snapshots}
        report = {
            "experiment": EXPERIMENT,
            "query_id": qid,
            "rescue_query": bool(plans[q["query"]].get("rescue_query")),
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "api_calls": telemetry.api_calls,
            "cache_hits": telemetry.cache_hits,
            "llm_calls": telemetry.llm_calls,
            "input_tokens": telemetry.input_tokens,
            "output_tokens": telemetry.output_tokens,
            "latency_ms": latency_ms,
            "n_predicted": len(results),
            "raw_candidates": snapshots.get("after_raw_recall", {}).get("candidate_count", 0),
            "rerank_pool_candidates": snapshots.get("rerank_pool", {}).get("candidate_count", 0),
        }
        recorder.save(TRACE_DIR)
        cache.flush()
        (M3_DIR / f"report_{qid}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        kind = "RESCUE" if report["rescue_query"] else "rich"
        print(f"[{i}/{len(todo)}] {qid} [{kind}]: F1={report['f1']:.4f} P={report['precision']:.4f} "
              f"R={report['recall']:.4f} api={report['api_calls']} cache={report['cache_hits']} "
              f"llm={report['llm_calls']} raw={report['raw_candidates']} {latency_ms}ms")

    cache.close()
    print("检索完成。生成产物...")
    cmd_report()


# --------------------------------------------------------------------------
# Phase D：report —— 五份产物 + 决策（读 reports + traces，不联网）
# --------------------------------------------------------------------------
def load_report(qid: str) -> dict:
    return json.loads((M3_DIR / f"report_{qid}.json").read_text(encoding="utf-8"))


def load_trace(qid: str) -> dict:
    return json.loads((TRACE_DIR / f"trace_{qid}.json").read_text(encoding="utf-8"))


def subquery_intent(plans: dict[str, dict], query_text: str) -> tuple[str, str]:
    """first_seen_query(80 字符截断) -> (intent, kind)。kind ∈ rescue/rich/unknown。"""
    key = query_text[:80]
    for p in plans.values():
        for s in p["subs"]:
            if s["query_text"][:80] == key:
                kind = "rescue" if p.get("rescue_query") else "rich"
                return s["intent"], kind
    return "unknown", "unknown"


def gold_papers_by_subquery(trace: dict, rescue_subs: list[dict]) -> dict[str, set[str]]:
    """trace 里 rescue 子查询 first-seen 的 gold title_n，按子查询文本分组。"""
    by_q: dict[str, set[str]] = {s["query_text"]: set() for s in rescue_subs}
    for g in trace.get("gold_lifecycle", []):
        key = g.get("first_seen_query", "")[:80]
        for s in rescue_subs:
            if s["query_text"][:80] == key:
                by_q[s["query_text"]].add(g["title_n"])
                break
    return by_q


def cmd_report() -> None:
    plans = load_m3_plans()
    b0 = load_nocit_baseline()
    queries = select_queries()
    v2 = load_v2_plans(V2_PLAN_CACHE)
    recall = load_recall_cache_map()
    rescue_plans = load_rescue_plans()
    sparse, rich = classify_sparse_rich(v2)
    meta = json.loads(META_FILE.read_text(encoding="utf-8")) if META_FILE.exists() else {}

    # 只聚合有 report+trace 的 query
    reports: dict[str, dict] = {}
    for q in queries:
        rep = M3_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            reports[q["query_id"]] = json.loads(rep.read_text(encoding="utf-8"))

    # ---- 全 gold 生命周期聚合 ----
    all_lifecycle: list[dict] = []
    rescue_subs_by_q: dict[str, list[dict]] = {
        q["query_id"]: rescue_plans[q["query"]]["subs"] for q in queries if q["query"] in sparse}
    for q in queries:
        qid = q["query_id"]
        if qid not in reports:
            continue
        trc = load_trace(qid)
        is_rescue = q["query"] in sparse
        for g in trc.get("gold_lifecycle", []):
            intent, kind = subquery_intent(plans, g.get("first_seen_query", ""))
            row = {**g, "query_id": qid, "query_type": intent,
                   "source_kind": kind, "rescue_query": is_rescue}
            all_lifecycle.append(row)

    # ---- 15 条 sparse query 的 rescue-before（原始 v2 子查询在 recall cache 里的 gold）----
    # 离线计算：不联网，用 _pasa_recall_cache 评估原始 sparse 子查询命中 gold 的 title_n 集合
    gold_title_ns_by_q: dict[str, set[str]] = {}
    for q in queries:
        if q["query_id"] not in reports:
            continue
        gold_title_ns_by_q[q["query_id"]] = {
            norm_title(g["title"]) for g in q.get("gold", []) if g.get("title") and norm_title(g["title"])}
    rescue_before: dict[str, set[str]] = {}  # qid -> 原始 sparse 子查询 hit 的 gold title_n
    for q in queries:
        qid = q["query_id"]
        if q["query"] not in sparse or qid not in gold_title_ns_by_q:
            continue
        orig_sub = v2[q["query"]]["subs"][0]["query_text"] if v2[q["query"]]["subs"] else q["query"]
        found: set[str] = set()
        for ev in recall.get(orig_sub, []):
            if is_gold_by_title(ev, gold_title_ns_by_q[qid]):
                found.add(norm_title(ev.identity.title))
        rescue_before[qid] = found

    # ---- m3_sparse_query_metrics.csv ----
    sparse_metrics: list[dict] = []
    for q in queries:
        qid = q["query_id"]
        if q["query"] not in sparse or qid not in reports:
            continue
        trc = load_trace(qid)
        rescue_subs = rescue_subs_by_q[qid]
        by_q = gold_papers_by_subquery(trc, rescue_subs)
        rescue_after = {tn for s in rescue_subs for tn in by_q[s["query_text"]]}
        before = rescue_before.get(qid, set())
        new_gold = rescue_after - before
        # per-type new gold：new gold 中由该 intent 子查询 first-seen 的
        type_new = {t: 0 for t in RESCUE_INTENTS}
        for s in rescue_subs:
            intent = s["intent"]
            type_new[intent] += len({tn for tn in by_q[s["query_text"]] if tn in new_gold})
        n_calls = len(rescue_subs)
        sparse_metrics.append({
            "query_id": qid,
            "original_subquery_count": len(v2[q["query"]]["subs"]),
            "rescue_subquery_count": n_calls,
            "rescue_before_raw_gold": len(before),
            "rescue_after_raw_gold": len(rescue_after),
            "new_gold_papers": len(new_gold),
            "incremental_gold_per_call": round(len(new_gold) / n_calls, 3) if n_calls else 0.0,
            "core_new_gold": type_new[INTENT_CORE],
            "anchor_new_gold": type_new[INTENT_ANCHOR],
            "discovery_new_gold": type_new[INTENT_DISCOVERY],
        })
    with open(M3_DIR / "m3_sparse_query_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sparse_metrics[0].keys()) if sparse_metrics else
                           ["query_id"])
        w.writeheader()
        for row in sparse_metrics:
            w.writerow(row)

    # ---- 论文级 / 实例级 口径（M3 gate 用论文级 title_n）----
    raw_rows = [g for g in all_lifecycle if g.get("first_seen_stage") in ("regular_recall", "assoc_recall")]
    pool_rows = [g for g in all_lifecycle if g.get("survived_prekeep")]
    final_rows = [g for g in all_lifecycle if g.get("final_rank") is not None]
    raw_unique_gold_papers = len({g["title_n"] for g in raw_rows})
    raw_query_gold_instances = len({g["canonical_id"] for g in raw_rows})
    pool_unique_gold = len({g["title_n"] for g in pool_rows})
    final_unique_gold = len({g["title_n"] for g in final_rows})
    raw_to_pool = (pool_unique_gold / raw_unique_gold_papers) if raw_unique_gold_papers else 0.0
    pool_to_final = (final_unique_gold / pool_unique_gold) if pool_unique_gold else 0.0

    # 全 gold 生命周期（加 m3_new + gain_outcome 两列）
    for row in all_lifecycle:
        row["m3_new"] = 0
        row["gain_outcome"] = ""
        if row["rescue_query"] and row["query_id"] in rescue_before:
            if row["title_n"] not in rescue_before[row["query_id"]] and row["first_seen_stage"] in ("regular_recall", "assoc_recall"):
                row["m3_new"] = 1
                row["gain_outcome"] = ("RETRIEVAL_GAIN" if row.get("final_rank") is not None
                                       else "RETRIEVAL_GAIN_BUT_RERANK_LOSS")
    lc_cols = ["query_id", "canonical_id", "title_n", "first_seen_stage", "first_seen_query",
               "first_seen_rank", "query_type", "source_kind", "rescue_query", "m3_new",
               "gain_outcome", "survived_prekeep", "pre_rerank_rank", "reranker_rank",
               "final_rank", "drop_stage", "drop_reason"]
    with open(M3_DIR / "m3_gold_lifecycle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=lc_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_lifecycle:
            w.writerow(row)

    # ---- m3_vs_b0.csv ----
    n = len(reports)
    agg = {k: {"b0": 0.0, "m3": 0.0} for k in ("f1", "precision", "recall", "api_calls", "llm_calls")}
    total_phys = 0
    total_cache = 0
    total_latency = 0.0
    with open(M3_DIR / "m3_vs_b0.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "kind", "b0_f1", "b0_precision", "b0_recall", "b0_api_calls", "b0_llm_calls",
                "m3_f1", "m3_precision", "m3_recall", "m3_api_calls", "m3_cache_hits", "m3_llm_calls",
                "m3_raw_gold", "m3_final_gold", "delta_f1", "delta_recall"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for qid in sorted(reports):
            r = reports[qid]
            b = b0[qid]
            trc = load_trace(qid)
            total_phys += trc.get("physical_http_calls", r.get("api_calls", 0))
            total_cache += r.get("cache_hits", 0)
            total_latency += r.get("latency_ms", 0)
            raw_g = sum(1 for row in all_lifecycle if row["query_id"] == qid and
                        row["first_seen_stage"] in ("regular_recall", "assoc_recall"))
            fin_g = sum(1 for row in all_lifecycle if row["query_id"] == qid and row.get("final_rank") is not None)
            w.writerow({
                "query_id": qid, "kind": "RESCUE" if r.get("rescue_query") else "rich",
                "b0_f1": b["f1"], "b0_precision": b["precision"], "b0_recall": b["recall"],
                "b0_api_calls": b["api_calls"], "b0_llm_calls": b["llm_calls"],
                "m3_f1": r["f1"], "m3_precision": r["precision"], "m3_recall": r["recall"],
                "m3_api_calls": r["api_calls"], "m3_cache_hits": r.get("cache_hits", 0),
                "m3_llm_calls": r["llm_calls"],
                "m3_raw_gold": raw_g, "m3_final_gold": fin_g,
                "delta_f1": round(r["f1"] - b["f1"], 4), "delta_recall": round(r["recall"] - b["recall"], 4),
            })
            for k in agg:
                agg[k]["b0"] += b[k]
                agg[k]["m3"] += r[k]
        w.writerow({})
        w.writerow({
            "query_id": "MEAN", "kind": "",
            "b0_f1": round(agg["f1"]["b0"] / n, 4), "b0_precision": round(agg["precision"]["b0"] / n, 4),
            "b0_recall": round(agg["recall"]["b0"] / n, 4), "b0_api_calls": round(agg["api_calls"]["b0"] / n, 1),
            "b0_llm_calls": round(agg["llm_calls"]["b0"] / n, 1),
            "m3_f1": round(agg["f1"]["m3"] / n, 4), "m3_precision": round(agg["precision"]["m3"] / n, 4),
            "m3_recall": round(agg["recall"]["m3"] / n, 4), "m3_api_calls": round(agg["api_calls"]["m3"] / n, 1),
            "m3_cache_hits": round(total_cache / n, 1), "m3_llm_calls": round(agg["llm_calls"]["m3"] / n, 1),
            "m3_raw_gold": raw_unique_gold_papers, "m3_final_gold": final_unique_gold,
            "delta_f1": round(agg["f1"]["m3"] / n - agg["f1"]["b0"] / n, 4),
            "delta_recall": round(agg["recall"]["m3"] / n - agg["recall"]["b0"] / n, 4),
        })

    # ---- m3_decision.md ----
    gain = sum(1 for row in all_lifecycle if row.get("gain_outcome") == "RETRIEVAL_GAIN")
    loss = sum(1 for row in all_lifecycle if row.get("gain_outcome") == "RETRIEVAL_GAIN_BUT_RERANK_LOSS")
    rescue_new_total = sum(sm["new_gold_papers"] for sm in sparse_metrics)
    lines = []
    lines.append("# M3_SPARSE_PLAN_RESCUE 决策报告")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}（M3 全量 {n}/{len(queries)}-query 统一评测）")
    lines.append(f"- M3_PLANNER_VERSION={M3_PLANNER_VERSION}，plan_hash={meta.get('plan_hash', '?')[:16]}…")
    lines.append(f"- 对照 B0：`{BASELINE_REF}`（F1=0.0618 P=0.0615 R=0.1339 api=87 llm=77）——未重跑，读基线报告")
    lines.append(f"- M3 实验：`{EXPERIMENT}`。**7 条 rich plan 冻结原样**（含 v2 assoc 锚点），"
                 f"**15 条 sparse plan 用 SparsePlanRescue 重写**（≤4：core×1+anchor×1-2+discovery×0-1）。")
    lines.append("- 冻结：assoc_safepass=True（KEEP_SAFEPASS）、Citation/Reference/Metadata OFF、"
                 "OpenAlex 不变、top_k=20、Prekeep 不变、Reranker 不变。")
    lines.append("- 缓存/网络：rich/既有子查询全读 _pasa_recall_cache（0 联网）；仅新 rescue 子查询请求 OpenAlex。")
    lines.append("")
    lines.append("## 1. 检索原始召回（论文级 unique Gold，title_n 去重；M3 gate 口径）")
    lines.append("")
    lines.append(f"- **raw_unique_gold_papers = {raw_unique_gold_papers}**（B0 基线 22；15 条 sparse 原贡献 ~0）")
    lines.append(f"- **raw_query_gold_instances = {raw_query_gold_instances}**（canonical_id/DOI 实例级）")
    lines.append(f"- pool_unique_gold = {pool_unique_gold}（survived_prekeep，论文级）")
    lines.append(f"- final_unique_gold = {final_unique_gold}（final top-{TOP_K}，论文级）")
    lines.append(f"- raw→pool = {raw_to_pool:.1%}，pool→final = {pool_to_final:.1%}")
    lines.append("")
    lines.append("## 2. Rescue 增益（15 条 sparse query）")
    lines.append("")
    lines.append(f"- rescue 新增 unique Gold 论文（未命中原始 sparse plan 的）：**{rescue_new_total}**")
    lines.append(f"- 其中 RETRIEVAL_GAIN（新 Gold 最终进 top-k）= **{gain}**；"
                 f"RETRIEVAL_GAIN_BUT_RERANK_LOSS（raw 命中原 Gold 但被 prekeep/reranker 丢弃）= **{loss}**")
    lines.append("")
    lines.append("| query_id | rescue前 | rescue后 | new | CORE | ANCHOR | DISCOVERY | calls | 增量/call |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sm in sparse_metrics:
        lines.append(f"| {sm['query_id']} | {sm['rescue_before_raw_gold']} | {sm['rescue_after_raw_gold']} | "
                     f"{sm['new_gold_papers']} | {sm['core_new_gold']} | {sm['anchor_new_gold']} | "
                     f"{sm['discovery_new_gold']} | {sm['rescue_subquery_count']} | {sm['incremental_gold_per_call']} |")
    lines.append("")
    lines.append("## 3. 核心指标（vs B0）")
    lines.append("")
    lines.append("| 指标 | B0 | M3 | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| F1 | {agg['f1']['b0']/n:.4f} | {agg['f1']['m3']/n:.4f} | {agg['f1']['m3']/n - agg['f1']['b0']/n:+.4f} |")
    lines.append(f"| Precision | {agg['precision']['b0']/n:.4f} | {agg['precision']['m3']/n:.4f} | {agg['precision']['m3']/n - agg['precision']['b0']/n:+.4f} |")
    lines.append(f"| Recall | {agg['recall']['b0']/n:.4f} | {agg['recall']['m3']/n:.4f} | {agg['recall']['m3']/n - agg['recall']['b0']/n:+.4f} |")
    lines.append(f"| logical API calls/query | {agg['api_calls']['b0']/n:.1f} | {agg['api_calls']['m3']/n:.1f} | |")
    lines.append(f"| physical HTTP attempts（全量） | 87 | {total_phys} | |")
    lines.append(f"| LLM calls/query | {agg['llm_calls']['b0']/n:.1f} | {agg['llm_calls']['m3']/n:.1f} | |")
    lines.append(f"| 总延迟（ms） | - | {total_latency:.0f} | |")
    lines.append("")
    lines.append("## 4. 决策 Gate（论文级 raw，基线 22/180）")
    lines.append("")
    lines.append("- ≥40 → STRONG_RETRIEVAL_SUCCESS（立即冻结 M3 Planner）；≥35 → MVP_RETRIEVAL_SUCCESS；"
                 "≥30 → QUERY_FORMULATION_SIGNAL；<30 → QUERY_FORMULATION_INSUFFICIENT（只允许 1 次 prompt 修订）。")
    if raw_unique_gold_papers >= 40:
        gate = "STRONG_RETRIEVAL_SUCCESS"
    elif raw_unique_gold_papers >= 35:
        gate = "MVP_RETRIEVAL_SUCCESS"
    elif raw_unique_gold_papers >= 30:
        gate = "QUERY_FORMULATION_SIGNAL"
    else:
        gate = "QUERY_FORMULATION_INSUFFICIENT"
    lines.append(f"- **判定：{gate}**（raw_unique_gold_papers = {raw_unique_gold_papers}）")
    lines.append("")
    if raw_unique_gold_papers >= 35:
        if final_unique_gold / raw_unique_gold_papers < 0.5:
            lines.append("**NEXT = M4_RERANKER_RETENTION**：raw≥35 但 final 受限（raw→final 保留率 <50%），"
                         "下一步做 Gold-Retention Reranker（三剑第三剑）。")
        else:
            lines.append("raw 与 final 同向提升，可考虑 top-k 或精度截断优化。")
    elif raw_unique_gold_papers >= 30:
        lines.append("有信号但未达 MVP：可做 1 次 M3 Planner prompt 修订（只允许一次）。")
    else:
        lines.append("raw 未达标：只允许 1 次 M3 Planner prompt 修订，若仍不达标回 B0 基线。")
    lines.append("")
    lines.append("## 5. 范围遵守声明")
    lines.append("")
    lines.append("- src/search.py 未改动；src/planner.py 未改动。新增 src/planner_rescue.py + scripts/run_m3_sparse.py。")
    lines.append("- Gold isolation：rescue planner 输入仅 question 文本；严禁基于 Gold 生成论文标题/作者/DOI/arXiv ID。")
    lines.append("- 只解决 15 条 sparse plan 的 Query Formulation；7 条 rich plan 冻结原样，未重新调用。")
    lines.append(f"- 联网仅新 rescue 子查询（rescue_new_total 对应子查询），既有 v2 子查询 0 联网重跑。")
    lines.append("")
    lines.append("**本轮（M3）到此为止：完成 22 条统一评测后 STOP，等待用户批准后才进入 M4，不自行进入。**")
    (M3_DIR / "m3_decision.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== M3 汇总 ===")
    print(f"F1 {agg['f1']['b0']/n:.4f} -> {agg['f1']['m3']/n:.4f}")
    print(f"P  {agg['precision']['b0']/n:.4f} -> {agg['precision']['m3']/n:.4f}")
    print(f"R  {agg['recall']['b0']/n:.4f} -> {agg['recall']['m3']/n:.4f}")
    print(f"raw_unique_gold_papers={raw_unique_gold_papers} (instances={raw_query_gold_instances})  "
          f"pool={pool_unique_gold}  final={final_unique_gold}")
    print(f"rescue 新增 Gold={rescue_new_total}（RETRIEVAL_GAIN={gain}，RERANK_LOSS={loss}）")
    print(f"Gate：{gate}")
    print("产物：m3_sparse_query_metrics.csv / m3_gold_lifecycle.csv / m3_vs_b0.csv / m3_decision.md")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["generate", "build", "search", "report", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="跳过配额预检强制联网（不推荐，仅测试子集时用）")
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
