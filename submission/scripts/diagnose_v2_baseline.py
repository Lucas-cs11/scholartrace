"""Offline diagnostic report for the frozen v2 PASA_ASSOC baseline.

This script never calls a planner or an external API. It only reads the v2 plan,
recall cache, and per-query reports, and explicitly marks unavailable funnel
stages as null rather than inferring them from final scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

ASSOC_INTENT = "联想论文名"

def norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def title_keys(report: dict) -> set[str]:
    return {x.removeprefix("title_n:") for x in report["gold_ids"] if x.startswith("title_n:")}


def gold_count(report: dict) -> int:
    return sum(x.startswith("title:") for x in report["gold_ids"])


def candidate_key(ev: dict) -> str:
    ident = ev["identity"]
    return ident.get("paper_id") or ident.get("doi") or norm_title(ident.get("title", ""))


def candidate_title(ev: dict) -> str:
    return norm_title(ev["identity"].get("title", ""))


def hit_keys(evs: list[dict], gold: set[str]) -> set[str]:
    return {candidate_key(ev) for ev in evs if candidate_title(ev) in gold}


def union_evs(cache: dict[str, list[dict]], terms: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for term in terms:
        for ev in cache.get(term, []):
            out.setdefault(candidate_key(ev), ev)
    return out


def mean_overlap(term_sets: list[set[str]]) -> float | None:
    if len(term_sets) < 2:
        return None
    vals = []
    for a, b in combinations(term_sets, 2):
        denom = len(a | b)
        vals.append(len(a & b) / denom if denom else 0.0)
    return round(sum(vals) / len(vals), 4)


def match_trace_calls(report: dict, regular: list[str], assoc: list[str]) -> tuple[int, int, int]:
    """Classify round-1 traces by consuming duplicate subquery text in plan order."""
    remaining = [(q, "regular") for q in regular] + [(q, "assoc") for q in assoc]
    reg = assoc_n = raw = 0
    for trace in report.get("traces", []):
        if trace["round"] != 1:
            continue
        raw += 1
        query = trace["query"]
        for i, (text, kind) in enumerate(remaining):
            if text == query:
                remaining.pop(i)
                if kind == "regular":
                    reg += 1
                else:
                    assoc_n += 1
                break
    return raw, reg, assoc_n


def final_gold_ranks(report: dict, cache_by_id: dict[str, dict], gold: set[str]) -> list[int]:
    ranks = []
    for i, pid in enumerate(report.get("predicted_ids", []), 1):
        ev = cache_by_id.get(pid)
        if ev and candidate_title(ev) in gold:
            ranks.append(i)
    return ranks


def analyze(plan_path: Path, recall_path: Path, report_paths: list[Path]) -> tuple[list[dict], list[dict], dict]:
    plans = {d["query"]: d for d in load_jsonl(plan_path)}
    recall_rows = load_jsonl(recall_path)
    recall = {d["q"]: d["evs"] for d in recall_rows}
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in report_paths]
    if len(plans) != len(reports):
        raise ValueError(f"plan/report count mismatch: {len(plans)} != {len(reports)}")
    missing = [r["raw_query"] for r in reports if r["raw_query"] not in plans]
    if missing:
        raise ValueError(f"missing v2 plans for {len(missing)} reports")

    rows = []
    assoc_rows = []
    curves = []
    for report in reports:
        plan = plans[report["raw_query"]]
        if plan.get("v") != 2 or not isinstance(plan.get("ir"), dict) or not isinstance(plan.get("subs"), list):
            raise ValueError(f"incomplete or non-v2 plan: {report['query_id']}")
        regular = [s["query_text"] for s in plan["subs"] if s.get("intent") != ASSOC_INTENT]
        assoc = [s["query_text"] for s in plan["subs"] if s.get("intent") == ASSOC_INTENT]
        if not regular and not assoc:
            raise ValueError(f"empty plan: {report['query_id']}")
        gold = title_keys(report)
        reg_evs = union_evs(recall, regular)
        assoc_evs = union_evs(recall, assoc)
        raw_evs = {**reg_evs, **assoc_evs}
        raw_hits = hit_keys(list(raw_evs.values()), gold)
        regular_hits = hit_keys(list(reg_evs.values()), gold)
        assoc_hits = hit_keys(list(assoc_evs.values()), gold)
        cache_by_id = {candidate_key(ev): ev for evs in recall.values() for ev in evs}
        final_hits = {
            pid for pid in report.get("predicted_ids", [])
            if pid in cache_by_id and candidate_title(cache_by_id[pid]) in gold
        }
        raw_calls, regular_calls, assoc_calls = match_trace_calls(report, regular, assoc)
        citation_calls = max(0, report["api_calls"] - raw_calls)
        plan_hash = __import__("hashlib").sha256(
            json.dumps({"v": plan["v"], "ir": plan["ir"], "subs": plan["subs"]}, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        row = {
            "query_id": report["query_id"], "plan_hash": plan_hash,
            "baseline_F1": report["f1"], "gold_paper_count": gold_count(report),
            "regular_subquery_count": len(regular), "assoc_subquery_count": len(assoc),
            "regular_subqueries": json.dumps(regular, ensure_ascii=False),
            "assoc_subqueries": json.dumps(assoc, ensure_ascii=False),
            "raw_recall_api_calls": raw_calls, "regular_recall_api_calls": regular_calls,
            "assoc_recall_api_calls": assoc_calls, "citation_api_calls": citation_calls,
            "other_api_calls": max(0, report["api_calls"] - raw_calls - citation_calls),
            "total_api_calls": report["api_calls"],
            "raw_recall_unique_papers": len(raw_evs), "post_dedup_unique_papers": len(raw_evs),
            "post_citation_unique_papers": None, "gold_hit_after_raw_recall": len(raw_hits),
            "gold_hit_added_by_assoc": len(assoc_hits - regular_hits),
            "gold_hit_added_by_citation": None, "gold_hit_before_rerank": None,
            "gold_hit_after_rerank": len(final_hits), "final_gold_hit": len(final_hits),
            "gold_paper_rank": json.dumps(final_gold_ranks(report, cache_by_id, gold)),
            "citation_analysis_available": False,
        }
        rows.append(row)

        term_sets = []
        for term in assoc:
            evs = recall.get(term, [])
            keys = {candidate_key(ev) for ev in evs}
            term_sets.append(keys)
            assoc_rows.append({
                "query_id": report["query_id"], "term": term,
                "api_cost": 1 if term in recall else 0,
                "unique_papers": len(keys), "gold_papers": len(hit_keys(evs, gold)),
                "gold_papers_incremental_vs_regular": len(hit_keys(evs, gold) - regular_hits),
                "overlap_with_other_assoc_terms": mean_overlap([keys] + [x for x in term_sets[:-1]]),
            })
        # Raw execution order curve; citation is one aggregate trace and is explicitly incomplete.
        seen: dict[str, dict] = {}
        for trace in report.get("traces", []):
            if trace["round"] != 1:
                continue
            for ev in recall.get(trace["query"], []):
                seen.setdefault(candidate_key(ev), ev)
            curves.append({"query_id": report["query_id"], "step": len([x for x in curves if x["query_id"] == report["query_id"]]) + 1,
                           "stage": "raw_recall", "api": trace["query"], "cumulative_unique_papers": len(seen),
                           "cumulative_gold_hits": len(hit_keys(list(seen.values()), gold)),
                           "candidate_delta_reported": trace.get("candidate_delta", 0)})

    summary = {
        "queries": len(rows), "f1_zero_queries": sum(r["baseline_F1"] == 0 for r in rows),
        "f1_zero_no_raw_gold": sum(r["baseline_F1"] == 0 and r["gold_hit_after_raw_recall"] == 0 for r in rows),
        "f1_zero_gold_reached_rerank_unknown": None,
        "queries_assoc_added_gold": sum(r["gold_hit_added_by_assoc"] > 0 for r in rows),
        "queries_citation_added_gold": None,
        "mean_api_calls": round(sum(r["total_api_calls"] for r in rows) / len(rows), 2),
        "mean_f1": round(sum(r["baseline_F1"] for r in rows) / len(rows), 4),
        "citation_counterfactual": "unavailable: citation candidate payloads are not persisted in v2 artifacts",
        "rerank_funnel": "unavailable: pre-rerank candidate pool is not persisted in RunReport",
    }
    return rows, assoc_rows, {"summary": summary, "curves": curves}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--recall", type=Path, required=True)
    ap.add_argument("--reports", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report_paths = sorted(args.reports.glob("PASA_ASSOC_RealScholarQuery_*.json"))
    rows, assoc_rows, extra = analyze(args.plans, args.recall, report_paths)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "query_diagnostics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "assoc_terms.json").write_text(json.dumps(assoc_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "retrieval_curves.json").write_text(json.dumps(extra["curves"], ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(extra["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with (args.out / "query_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(extra["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
