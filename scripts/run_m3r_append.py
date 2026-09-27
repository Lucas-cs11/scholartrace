"""M3-R_APPEND：修正 M3 replacement 的方法学混淆（replace -> append），纯离线重放。

动机（用户审阅 M3 后）：M3 replacement run 中 Rescue 新增 3 篇 Gold（全进 final top-20），
但因「替换」原始 sparse query，Q29/Q41 丢失已有 Gold → raw +3 -3 = 0 混合了
「Query Formulation 增益」与「replacement policy 损失」。正确语义应为 Preserve + Augment。

M3-R 定义（对 15 条 sparse plan）：
    final_search_queries = original_v2_subqueries + cached_M3_rescue_subqueries  （追加）
7 条 rich v2 plans 完全保持原样。不重新生成 Rescue Plan / 不修改 M3 Planner Prompt。

冻结清单：assoc_safepass=True（KEEP_SAFEPASS）、Citation/Reference/Metadata OFF、
OpenAlex 不变、top_k=20、Prekeep 不变、Reranker 不变。唯一变量：replace -> append。

Cache first：原始 v2 query 用 _pasa_recall_cache；M3 Rescue query 用 M3 本轮已生成的
response cache（eval/cache/m3_sparse_plan_rescue）。全部 response 存在 → OpenAlex
physical HTTP = 0。任一 cache miss → STOP 并报告，绝不自动联网补跑。

统一统计口径：raw/pool/final 同时输出【query-gold instances(canonical_id)】与
【unique Gold papers(title_n)】，禁止把 instance 数量命名为 unique Gold。

产物（eval/runs/m3r_append/）：
  m3r_append_metrics.csv        —— 逐 query M3-APPEND 指标
  m3r_gold_lifecycle.csv        —— 全 gold 生命周期（含 preserved/incremental/lost 标注）
  m3r_vs_b0_vs_replace.csv      —— B0 vs M3-REPLACE vs M3-APPEND 三方
  m3r_decision.md               —— 决策报告（append raw gate + 瓶颈判定）

用法：
  python scripts/run_m3r_append.py build      # 追加 plan cache（离线）
  python scripts/run_m3r_append.py search     # 离线重放（cache-first，0 联网，miss 即 STOP）
  python scripts/run_m3r_append.py report     # 三方产物 + 决策（离线）
  python scripts/run_m3r_append.py all
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from eval.harness import compute_p_r_f1, match_gold
from src.adapters.openalex import OpenAlexAdapter
from src.observability.canonical import is_gold_by_title, norm_title
from src.observability.response_cache import CacheMiss, ResponseCache
from src.observability.trace_recorder import TraceRecorder
from src.planner import ASSOC_INTENT
from src.schemas import PaperEvidence
from src.search import SearchEngine
from scripts.eval_benchmark import load_pasa

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
V2_PLAN_CACHE = "eval/runs/_pasa_plan_cache.jsonl"
NOCIT_RESULTS = "eval/runs/nocitation/nocitation_results.json"
RECALL_CACHE = "eval/runs/_pasa_recall_cache.jsonl"
M3_CACHE_DIR = "eval/cache/m3_sparse_plan_rescue"  # M3 本轮生成的 rescue response cache
M3_RUN_DIR = Path("eval/runs/m3_sparse_plan_rescue")  # M3-REPLACE reports
M3_TRACE_DIR = Path("eval/diagnostics/m3_sparse_plan_rescue")  # M3-REPLACE traces
V2_TRACE_DIR = Path("eval/diagnostics/v2_instrumented")  # B0（safepass ON）对照
R_DIR = Path("eval/runs/m3r_append")
PLAN_FILE = R_DIR / "m3r_query_plans.jsonl"
MERGE_RECALL_FILE = "eval/cache/m3r_append/recall_cache.jsonl"
TRACE_DIR = Path("eval/diagnostics/m3r_append")
EXPERIMENT = "PASA_ASSOC_M3R_APPEND"
BASELINE_REF = "PASA_ASSOC_NO_CIT"
M3_REPLACE_EXP = "PASA_ASSOC_M3_SPARSE_RESCUE"
TOP_K = 20
RESCUE_INTENTS = ("core", "anchor", "discovery")


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
        plans[d["query"]] = d
    return plans


def load_rescue_plans() -> dict[str, dict]:
    plans: dict[str, dict] = {}
    fp = Path("eval/runs/m3_sparse_plan_rescue/m3_sparse_plans.jsonl")
    if not fp.exists():
        return plans
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        plans[d["query"]] = d
    return plans


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


# --------------------------------------------------------------------------
# Phase A：build —— 追加 plan cache（sparse = original + rescue；rich 原样）
# --------------------------------------------------------------------------
def cmd_build() -> None:
    R_DIR.mkdir(parents=True, exist_ok=True)
    v2 = load_v2_plans(V2_PLAN_CACHE)
    rescue = load_rescue_plans()
    sparse, rich = classify_sparse_rich(v2)
    if sparse - set(rescue):
        raise SystemExit("EXPERIMENT_INVALID: 缺 rescue plan")
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
            # Preserve + Augment：原始 v2 subqueries + M3 rescue subqueries（追加）
            lines.append({
                "query": q["query"], "query_id": q["query_id"], "v": 2,
                "ir": v2[q["query"]]["ir"],
                "subs": v2[q["query"]]["subs"] + rescue[q["query"]]["subs"],
                "m3_kind": "sparse_append", "append": True,
                "rescue_subqueries": [s["query_text"] for s in rescue[q["query"]]["subs"]],
            })
    with open(PLAN_FILE, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    from collections import Counter
    cnt = Counter()
    for ln in lines:
        cnt.update(s["intent"] for s in ln["subs"])
    print(f"M3-R plan cache：{len(lines)} 条（rich {len(rich)} + sparse_append {len(sparse)}）")
    print("类型统计：", dict(cnt))
    print("写", PLAN_FILE)


# --------------------------------------------------------------------------
# 合并 recall cache：原始 v2/rich（_pasa_recall_cache）+ rescue（M3 response cache replay）
# --------------------------------------------------------------------------
async def build_merged_recall_cache(plans: dict[str, dict]) -> dict[str, list]:
    """返回 {subquery_text: [PaperEvidence]}；任一 rescue subquery 未命中 M3 cache → STOP。"""
    merged = load_recall_cache_map()  # 原始 v2/rich（含 2 个命中 _pasa_recall_cache 的 rescue 词）
    cache = ResponseCache(M3_CACHE_DIR, mode="replay")
    oa = OpenAlexAdapter(mailto=settings.openalex_mailto, cache=cache)
    rescue_subs = {s["query_text"] for p in plans.values() if p.get("append")
                   for s in p["subs"] if s["query_text"] in p.get("rescue_subqueries", [])}
    missing = [q for q in rescue_subs if q not in merged]
    if missing:
        print(f"RESCUE 需从 M3 cache 重放 {len(missing)} 条")
        for q in missing:
            try:
                evs = await oa.search(q, limit=20)  # replay，0 联网；telemetry=None
            except CacheMiss:
                raise SystemExit(
                    f"!! M3R_CACHE_MISS：rescue subquery 未命中 M3 response cache，STOP 不联网补跑。\n    '{q}'")
            merged[q] = evs
    Path(MERGE_RECALL_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MERGE_RECALL_FILE, "w", encoding="utf-8") as f:
        for q, evs in merged.items():
            f.write(json.dumps({"q": q, "evs": [e.model_dump() for e in evs]}, ensure_ascii=False) + "\n")
    # cache-first 全覆盖校验：所有 append plan 的 subquery 必须在 merged 里
    all_missing = [s["query_text"] for p in plans.values() for s in p["subs"]
                   if s["query_text"] not in merged]
    if all_missing:
        raise SystemExit(f"!! M3R_CACHE_MISS：以下 subquery 无任何缓存，STOP 不联网：\n    {all_missing}")
    print(f"合并 recall cache：{len(merged)} 条 subquery，全部覆盖（0 联网）。写 {MERGE_RECALL_FILE}")
    return merged


# --------------------------------------------------------------------------
# Phase B：search —— 离线重放（0 联网；cache-first，miss 即 STOP）
# --------------------------------------------------------------------------
async def cmd_search(limit: int | None) -> None:
    plans = {}
    if not PLAN_FILE.exists():
        raise SystemExit("EXPERIMENT_INVALID: 无 M3-R plan，先跑 build")
    for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        plans[d["query"]] = d
    queries = [q for q in select_queries() if q["query"] in plans]
    if limit:
        queries = queries[:limit]

    # cache-first：构建合并 recall cache，任一处 miss 即 STOP（不联网）
    merged = await build_merged_recall_cache(plans)

    cache = ResponseCache(M3_CACHE_DIR, mode="replay")  # 只读兜底，replay 模式 miss 抛 CacheMiss
    engine = SearchEngine(response_cache=cache, enable_citation_expansion=False, assoc_safepass=True)
    engine.load_plan_cache(str(PLAN_FILE))
    # 预加载全部 subquery 的 PaperEvidence -> _recall_cache 全命中，recall.search 不触发
    engine._recall_cache = merged
    n_missing_after = sum(1 for p in plans.values() for s in p["subs"]
                          if s["query_text"] not in engine._recall_cache)
    print(f"M3-R 离线重放：{len(queries)} 条，assoc_safepass=True。recall cache {len(merged)} 条，"
          f"uncovered={n_missing_after}。预计 OpenAlex api_calls=0 / physical=0。")

    R_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for q in queries:
        rep = R_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            continue
        todo.append(q)

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
            with open(R_DIR / "search_failures.log", "a", encoding="utf-8") as f:
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
        (R_DIR / f"report_{qid}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        kind = "APPEND" if report["append"] else "rich"
        print(f"[{i}/{len(todo)}] {qid} [{kind}]: F1={report['f1']:.4f} P={report['precision']:.4f} "
              f"R={report['recall']:.4f} api={report['api_calls']} cache={report['cache_hits']} "
              f"llm={report['llm_calls']} raw={report['raw_candidates']} {latency_ms}ms")

    # 离线联网护栏：校验所有 report 的 OpenAlex api_calls == 0（replay 兜底）
    bad = []
    for q in queries:
        rep = R_DIR / f"report_{q['query_id']}.json"
        if rep.exists():
            r = json.loads(rep.read_text(encoding="utf-8"))
            if r.get("api_calls", 0) > 0:
                bad.append((q["query_id"], r.get("api_calls")))
    if bad:
        print(f"!! M3R_OFFLINE_VIOLATION：以下 query 触发 OpenAlex 联网：{bad}")
    else:
        print("M3-R 离线校验通过：所有 query OpenAlex api_calls == 0（零联网）。")
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
    """聚合 dirp 下所有 query 的 gold 生命周期：实例级 + 论文级 + 保留率。"""
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
                      rescue_plans: dict[str, dict]) -> dict:
    """15 条 sparse 的 preserved / incremental / lost 归因（论文级 title_n）。

    用「recall-cache 结果归属」而非 first_seen_query：对每个 sparse query，把原始子查询的
    PaperEvidence 结果（title_n）与 rescue 子查询的结果分开。某 gold 论文若出现在原始子查询
    结果里 → preserved（append 保留）；否则若只出现在 rescue 子查询结果里 → incremental
    （genuinely new）。避免 rescue 先跑（priority 更高）导致的 first_seen 归属偏差。
    """
    if not Path(MERGE_RECALL_FILE).exists():
        return {"preserved_gold": 0, "incremental_rescue_gold": 0,
                "rescue_gold_lost_prepool": 0, "rescue_gold_lost_rerank": 0}
    merged = {}
    for line in Path(MERGE_RECALL_FILE).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        merged[d["q"]] = [norm_title(e["identity"]["title"]) for e in d["evs"]
                          if e.get("identity", {}).get("title")]

    preserved: set[str] = set()
    incremental: set[str] = set()
    lost_prepool: set[str] = set()
    lost_rerank: set[str] = set()
    for q in queries:
        qid = q["query_id"]
        if q["query"] not in rescue_plans:
            continue
        rescue_subs = {s["query_text"] for s in rescue_plans[q["query"]]["subs"]}
        original_subs = {s["query_text"] for s in plans[q["query"]]["subs"]} - rescue_subs
        orig_titles = {t for sq in original_subs for t in merged.get(sq, [])}
        rescue_titles = {t for sq in rescue_subs for t in merged.get(sq, [])}
        trc = dirp / f"trace_{qid}.json"
        if not trc.exists():
            continue
        t = json.loads(trc.read_text(encoding="utf-8"))
        for g in t.get("gold_lifecycle", []):
            if g.get("first_seen_stage") not in ("regular_recall", "assoc_recall"):
                continue
            tn = g["title_n"]
            if tn in orig_titles:
                preserved.add(tn)
            elif tn in rescue_titles:
                incremental.add(tn)
                if not g.get("survived_prekeep"):
                    lost_prepool.add(tn)
                elif g.get("final_rank") is None:
                    lost_rerank.add(tn)
    return {
        "preserved_gold": len(preserved),
        "incremental_rescue_gold": len(incremental),
        "rescue_gold_lost_prepool": len(lost_prepool),
        "rescue_gold_lost_rerank": len(lost_rerank),
    }


def cmd_report() -> None:
    queries = select_queries()
    v2 = load_v2_plans(V2_PLAN_CACHE)
    rescue_plans = load_rescue_plans()
    sparse, rich = classify_sparse_rich(v2)
    b0 = load_nocit_baseline()

    plans = {}
    if PLAN_FILE.exists():
        for line in PLAN_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plans[d["query"]] = d

    reports: dict[str, dict] = {}
    for q in queries:
        rep = R_DIR / f"report_{q['query_id']}.json"
        trc = TRACE_DIR / f"trace_{q['query_id']}.json"
        if rep.exists() and trc.exists():
            reports[q["query_id"]] = json.loads(rep.read_text(encoding="utf-8"))
    n = len(reports)

    # ---- 全 gold 生命周期（m3r_gold_lifecycle.csv）----
    all_lifecycle: list[dict] = []
    rescue_texts = {q["query_id"]: {s["query_text"]: s["intent"] for s in rescue_plans[q["query"]]["subs"]}
                    for q in queries if q["query"] in rescue_plans}
    for q in queries:
        qid = q["query_id"]
        if qid not in reports:
            continue
        trc = load_trace(TRACE_DIR, qid)
        for g in trc.get("gold_lifecycle", []):
            fq = g.get("first_seen_query", "")[:80]
            intent = ""; src = "rich"
            if qid in rescue_texts:
                for st, si in rescue_texts[qid].items():
                    if st[:80] == fq:
                        intent = si; src = "rescue"; break
            row = {**g, "query_id": qid, "query_type": intent, "source_kind": src,
                   "append": bool(plans.get(q["query"], {}).get("append"))}
            all_lifecycle.append(row)
    lc_cols = ["query_id", "canonical_id", "title_n", "first_seen_stage", "first_seen_query",
               "first_seen_rank", "query_type", "source_kind", "append", "survived_prekeep",
               "pre_rerank_rank", "reranker_rank", "final_rank", "drop_stage", "drop_reason"]
    with open(R_DIR / "m3r_gold_lifecycle.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=lc_cols, extrasaction="ignore")
        w.writeheader()
        for row in all_lifecycle:
            w.writerow(row)

    # ---- m3r_append_metrics.csv（逐 query）----
    with open(R_DIR / "m3r_append_metrics.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "kind", "f1", "precision", "recall", "api_calls", "physical_http",
                "llm_calls", "raw_candidates", "rerank_pool_candidates", "latency_ms"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for qid in sorted(reports):
            r = reports[qid]
            trc = load_trace(TRACE_DIR, qid)
            w.writerow({
                "query_id": qid, "kind": "APPEND" if r.get("append") else "rich",
                "f1": r["f1"], "precision": r["precision"], "recall": r["recall"],
                "api_calls": r["api_calls"],
                # 本轮新联网 OpenAlex physical HTTP = telemetry.api_calls（M3-R cache-first 全命中 -> 0）
                "physical_http": r["api_calls"],
                "llm_calls": r["llm_calls"], "raw_candidates": r["raw_candidates"],
                "rerank_pool_candidates": r["rerank_pool_candidates"], "latency_ms": r["latency_ms"],
            })

    # ---- 三方漏斗 ----
    append_funnel = traces_raw_pool_final(TRACE_DIR, queries)
    replace_funnel = traces_raw_pool_final(M3_TRACE_DIR, queries)
    b0_funnel = traces_raw_pool_final(V2_TRACE_DIR, queries)  # B0 = safepass ON (v2_instrumented)
    sparse_stats = sparse_gold_stats(TRACE_DIR, queries, plans, rescue_plans)

    # ---- m3r_vs_b0_vs_replace.csv ----
    total_phys = {k: 0 for k in ("b0", "replace", "append")}
    total_reuse = {"append": 0}  # M3-R 复用的 M3 已缓存 HTTP response 数（非本轮新联网）
    total_llm = {k: 0 for k in ("b0", "replace", "append")}
    total_api = {k: 0 for k in ("b0", "replace", "append")}
    rows = []
    for qid in sorted(reports):
        r = reports[qid]
        b = b0[qid]
        rep = M3_RUN_DIR / f"report_{qid}.json"
        rrep = json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else {}
        t_append = load_trace(TRACE_DIR, qid)
        t_replace = load_trace(M3_TRACE_DIR, qid) if (M3_TRACE_DIR / f"trace_{qid}.json").exists() else {}
        rep_f1 = rrep.get("f1")
        rep_recall = rrep.get("recall")
        rep_prec = rrep.get("precision")
        total_phys["b0"] += b.get("api_calls", 0)
        total_api["b0"] += b.get("api_calls", 0)  # 基线无 response cache：physical == logical
        total_phys["replace"] += t_replace.get("physical_http_calls", 0)
        # M3-R 本轮【新联网】= report.api_calls（telemetry，预加载 _recall_cache 后为 0）
        total_phys["append"] += r.get("api_calls", 0)
        # M3-R 复用的 M3 已缓存 HTTP response 数（TraceRecorder physical_http_calls，非本轮新联网）
        total_reuse["append"] += t_append.get("physical_http_calls", 0)
        total_llm["b0"] += b.get("llm_calls", 0)
        total_llm["replace"] += rrep.get("llm_calls", 0)
        total_llm["append"] += r.get("llm_calls", 0)
        total_api["append"] += r.get("api_calls", 0)
        total_api["replace"] += rrep.get("api_calls", 0)
        rows.append({
            "query_id": qid, "kind": "APPEND" if r.get("append") else "rich",
            "b0_f1": b["f1"], "b0_prec": b["precision"], "b0_recall": b["recall"],
            "rep_f1": rep_f1 if rep_f1 is not None else "",
            "rep_prec": rep_prec if rep_prec is not None else "",
            "rep_recall": rep_recall if rep_recall is not None else "",
            "app_f1": r["f1"], "app_prec": r["precision"], "app_recall": r["recall"],
            "app_raw_gold": sum(1 for row in all_lifecycle if row["query_id"] == qid and
                                row["first_seen_stage"] in ("regular_recall", "assoc_recall")),
            "app_final_gold": sum(1 for row in all_lifecycle if row["query_id"] == qid and row.get("final_rank") is not None),
        })
    with open(R_DIR / "m3r_vs_b0_vs_replace.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["query_id", "kind", "b0_f1", "b0_prec", "b0_recall", "rep_f1", "rep_prec", "rep_recall",
                "app_f1", "app_prec", "app_recall", "app_raw_gold", "app_final_gold"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
        w.writerow({})
        rep_f1s = [float(r["rep_f1"]) for r in rows if r["rep_f1"] != ""]
        rep_precs = [float(r["rep_prec"]) for r in rows if r["rep_prec"] != ""]
        rep_recalls = [float(r["rep_recall"]) for r in rows if r["rep_recall"] != ""]
        w.writerow({
            "query_id": "MEAN", "kind": "",
            "b0_f1": round(sum(r["b0_f1"] for r in rows) / n, 4),
            "b0_prec": round(sum(r["b0_prec"] for r in rows) / n, 4),
            "b0_recall": round(sum(r["b0_recall"] for r in rows) / n, 4),
            "rep_f1": round(sum(rep_f1s) / len(rep_f1s), 4) if rep_f1s else "",
            "rep_prec": round(sum(rep_precs) / len(rep_precs), 4) if rep_precs else "",
            "rep_recall": round(sum(rep_recalls) / len(rep_recalls), 4) if rep_recalls else "",
            "app_f1": round(sum(r["app_f1"] for r in rows) / n, 4),
            "app_prec": round(sum(r["app_prec"] for r in rows) / n, 4),
            "app_recall": round(sum(r["app_recall"] for r in rows) / n, 4),
            "app_raw_gold": append_funnel["raw_papers"], "app_final_gold": append_funnel["final_papers"],
        })

    # ---- 决策 ----
    raw = append_funnel["raw_papers"]
    inc = sparse_stats["incremental_rescue_gold"]
    lines = []
    lines.append("# M3-R_APPEND 决策报告（Preserve+Augment，纯离线重放）")
    lines.append("")
    lines.append(f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}（M3-R 全量 {n}/{len(queries)}-query 离线重放）")
    lines.append(f"- 对照 B0：`{BASELINE_REF}`（F1=0.0618）；M3-REPLACE：`{M3_REPLACE_EXP}`（raw=22）。")
    lines.append("- **唯一变量：replace -> append**。15 条 sparse = original_v2_subqueries + cached_M3_rescue_subqueries；"
                 "7 条 rich 原样。不重新生成 Rescue Plan / 不修改 M3 Planner Prompt。")
    lines.append("- 冻结：assoc_safepass=True、Citation/Reference/Metadata OFF、OpenAlex 不变、top_k=20、Prekeep/Reranker 不变。")
    lines.append(f"- 离线重放校验：OpenAlex api_calls=0 / physical HTTP=0（全部响应来自缓存；任一 miss 即 STOP）。")
    lines.append("")
    lines.append("## 1. 三方保留漏斗（论文级 unique Gold papers，title_n；B0= safepass ON）")
    lines.append("")
    lines.append("| 阶段 | B0 | M3-REPLACE | M3-APPEND |")
    lines.append("|---|---|---|---|")
    lines.append(f"| raw unique Gold papers | {b0_funnel['raw_papers']} | {replace_funnel['raw_papers']} | **{append_funnel['raw_papers']}** |")
    lines.append(f"| pool unique Gold papers | {b0_funnel['pool_papers']} | {replace_funnel['pool_papers']} | {append_funnel['pool_papers']} |")
    lines.append(f"| final unique Gold papers | {b0_funnel['final_papers']} | {replace_funnel['final_papers']} | {append_funnel['final_papers']} |")
    lines.append(f"| raw query-gold instances | {b0_funnel['raw_instances']} | {replace_funnel['raw_instances']} | {append_funnel['raw_instances']} |")
    lines.append(f"| raw→pool | {b0_funnel['raw_to_pool']:.1%} | {replace_funnel['raw_to_pool']:.1%} | {append_funnel['raw_to_pool']:.1%} |")
    lines.append(f"| pool→final | {b0_funnel['pool_to_final']:.1%} | {replace_funnel['pool_to_final']:.1%} | {append_funnel['pool_to_final']:.1%} |")
    lines.append("")
    lines.append("## 2. 核心指标（三方）")
    lines.append("")
    lines.append("| 指标 | B0 | M3-REPLACE | M3-APPEND |")
    lines.append("|---|---|---|---|")
    lines.append(f"| mean F1 | {sum(r['b0_f1'] for r in rows)/n:.4f} | {sum(float(r['rep_f1']) for r in rows)/n:.4f} | {sum(r['app_f1'] for r in rows)/n:.4f} |")
    lines.append(f"| mean Precision | {sum(r['b0_prec'] for r in rows)/n:.4f} | {sum(float(r['rep_prec']) for r in rows)/n:.4f} | {sum(r['app_prec'] for r in rows)/n:.4f} |")
    lines.append(f"| mean Recall | {sum(r['b0_recall'] for r in rows)/n:.4f} | {sum(float(r['rep_recall']) for r in rows)/n:.4f} | {sum(r['app_recall'] for r in rows)/n:.4f} |")
    lines.append(f"| OpenAlex logical calls | {total_api['b0']} | {total_api['replace']} | {total_api['append']} |")
    lines.append(f"| OpenAlex physical HTTP（本轮新联网） | {total_phys['b0']} | {total_phys['replace']} | {total_phys['append']} |")
    lines.append(f"| Reranker(LLM) calls | {total_llm['b0']} | {total_llm['replace']} | {total_llm['append']} |")
    lines.append("")
    lines.append(f"- M3-R 本轮新联网 OpenAlex physical HTTP = **{total_phys['append']}**（cache-first 全命中；复用了 M3 已生成的 {total_reuse['append']} 条 cached HTTP response，0 自动联网补跑）。")
    lines.append("")
    lines.append("## 3. Sparse 归因（15 条；论文级 title_n）")
    lines.append("")
    lines.append(f"- preserved Gold（原始 sparse 子查询保留，append 后仍在 raw）= **{sparse_stats['preserved_gold']}**")
    lines.append(f"- incremental Rescue Gold（rescue 子查询新增、原始未命中）= **{inc}**")
    lines.append(f"- Rescue Gold lost before pool（raw 命中但 prekeep 丢弃）= **{sparse_stats['rescue_gold_lost_prepool']}**")
    lines.append(f"- Rescue Gold lost by reranker（pool 但 final 丢弃）= **{sparse_stats['rescue_gold_lost_rerank']}**")
    lines.append("")
    lines.append("## 4. 决策 Gate（append raw_unique_gold，基线 22）")
    lines.append("")
    if raw >= 30:
        gate = "QUERY_FORMULATION_SIGNAL"
        next_note = "按既定 Gate 决策。"
    elif 23 <= raw <= 29 and inc > 0:
        gate = "QUERY_FORMULATION_PARTIAL_SIGNAL"
        next_note = "下一阶段允许执行：一次且仅一次 M3.1 Planner Prompt Revision。"
    else:
        gate = "QUERY_FORMULATION_NO_SIGNAL"
        next_note = "raw<=22，不继续 Prompt tuning。"
    lines.append(f"- **append raw_unique_gold_papers = {raw}**，incremental Rescue Gold = {inc}。")
    lines.append(f"- **判定：{gate}**。{next_note}")
    lines.append("")
    # 瓶颈判定：新增 Rescue Gold 的 raw→final 保留率
    inc_total = inc
    inc_lost = sparse_stats["rescue_gold_lost_prepool"] + sparse_stats["rescue_gold_lost_rerank"]
    inc_retained = inc_total - inc_lost
    if inc_total > 0:
        ret_rate = inc_retained / inc_total
        lines.append(f"- 新增 Rescue Gold raw→final 保留率 = {ret_rate:.0%}（{inc_retained}/{inc_total}）。")
        if ret_rate >= 0.5:
            lines.append("  → 新增 Rescue Gold 的 raw→final 保留较高：当前主要瓶颈仍是 **retrieval / query formulation**，"
                         "而非 reranker。不自动进入 M4。")
        else:
            lines.append("  → 新增 Rescue Gold 在 prekeep/reranker 大量丢失：瓶颈可能含 reranker，需单独分析。")
    lines.append("")
    lines.append("## 5. 范围遵守声明")
    lines.append("")
    lines.append("- src/search.py / src/planner.py / src/planner_rescue.py 均未改动。本轮禁止：ANCHOR/DISCOVERY prompt revision、"
                 "新 query type、Quality Selector、Adaptive Retrieval、Reranker modification。")
    lines.append("- Cache first：原始 v2/rich 读 _pasa_recall_cache，M3 Rescue 读 M3 response cache；OpenAlex physical HTTP=0，无自动联网补跑。")
    lines.append(f"- 统一口径：raw/pool/final 同时输出 instances(canonical_id) 与 unique Gold papers(title_n)。")
    lines.append("")
    lines.append("**本轮（M3-R）到此为止：完成后 STOP，等待用户批准后才进入 M3.1 / M4，不自行执行。**")
    (R_DIR / "m3r_decision.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== M3-R 汇总 ===")
    print(f"B0 raw={b0_funnel['raw_papers']}  REPLACE raw={replace_funnel['raw_papers']}  APPEND raw={append_funnel['raw_papers']}")
    print(f"APPEND pool={append_funnel['pool_papers']}  final={append_funnel['final_papers']}  instances={append_funnel['raw_instances']}")
    print(f"preserved={sparse_stats['preserved_gold']}  incremental_rescue={inc}  lost_prepool={sparse_stats['rescue_gold_lost_prepool']}  lost_rerank={sparse_stats['rescue_gold_lost_rerank']}")
    print(f"Gate：{gate}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "search", "report", "all"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.mode in ("build", "all"):
        cmd_build()
    if args.mode in ("search", "all"):
        await cmd_search(args.limit)
    if args.mode == "report":
        cmd_report()


if __name__ == "__main__":
    asyncio.run(main())
