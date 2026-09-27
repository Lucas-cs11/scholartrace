"""Offline counterfactual retrieval analysis for frozen v2 plans/caches.

No planner, reranker, or external API is called. Results are retrieval-level
counterfactuals; exact F1 is intentionally null because cached citation payloads
and counterfactual reranker outputs are not persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

ASSOC_INTENT = "联想论文名"

def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def key(ev: dict) -> str:
    ident = ev["identity"]
    return ident.get("paper_id") or ident.get("doi") or norm(ident.get("title", ""))

def title(ev: dict) -> str:
    return norm(ev["identity"].get("title", ""))

def gold(report: dict) -> set[str]:
    return {x[8:] for x in report["gold_ids"] if x.startswith("title_n:")}

def union(cache: dict[str, list[dict]], terms: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for term in terms:
        for ev in cache.get(term, []):
            result.setdefault(key(ev), ev)
    return result

def plan_hash(plan: dict) -> str:
    payload = {"v": plan["v"], "ir": plan["ir"], "subs": plan["subs"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def metrics(evs: dict[str, dict], gold_titles: set[str]) -> dict:
    hits = {key(ev) for ev in evs.values() if title(ev) in gold_titles}
    return {"unique_papers": len(evs), "gold_hits": len(hits), "gold_recall": round(len(hits) / len(gold_titles), 4) if gold_titles else 0.0}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, required=True)
    ap.add_argument("--recall", type=Path, required=True)
    ap.add_argument("--reports", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plans = {x["query"]: x for x in load_jsonl(args.plans)}
    cache = {x["q"]: x["evs"] for x in load_jsonl(args.recall)}
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.reports.glob("PASA_ASSOC_RealScholarQuery_*.json"))]
    if len(plans) != 22 or len(reports) != 22:
        raise SystemExit(f"expected 22 frozen v2 plans/reports, got {len(plans)}/{len(reports)}")

    curves = []
    per_query = []
    for report in reports:
        plan = plans.get(report["raw_query"])
        if not plan or plan.get("v") != 2 or not isinstance(plan.get("ir"), dict) or not plan.get("subs"):
            raise SystemExit(f"invalid/missing v2 plan for {report['query_id']}")
        regular = [s["query_text"] for s in plan["subs"] if s.get("intent") != ASSOC_INTENT]
        assoc = [s["query_text"] for s in plan["subs"] if s.get("intent") == ASSOC_INTENT]
        if any(term not in cache for term in regular + assoc):
            missing = [term for term in regular + assoc if term not in cache]
            raise SystemExit(f"missing recall cache for {report['query_id']}: {missing}")
        g = gold(report)
        reg = union(cache, regular)
        status = "error" if any(t.get("api") == "error" for t in report.get("traces", [])) else "ok"
        qrow = {"query_id": report["query_id"], "plan_hash": plan_hash(plan), "baseline_f1": report["f1"], "baseline_status": status}
        scenarios = []
        for count in range(len(assoc) + 1):
            evs = dict(reg)
            evs.update(union(cache, assoc[:count]))
            m = metrics(evs, g)
            scenarios.append({"assoc_count": count, **m, "estimated_raw_api_calls": len(regular) + count})
        best = []
        if len(assoc) >= 3:
            for combo in itertools.combinations(range(len(assoc)), 3):
                evs = dict(reg); evs.update(union(cache, [assoc[i] for i in combo]))
                m = metrics(evs, g)
                best.append((m["gold_hits"], m["unique_papers"], combo))
            best.sort(key=lambda x: (-x[0], x[1]))
        qrow.update({"retrieval_counterfactual": scenarios, "oracle_best3": [{"terms": [assoc[i] for i in combo], **metrics(dict(reg) | union(cache, [assoc[i] for i in combo]), g)} for _, _, combo in best[:1]]})
        per_query.append(qrow)
        curves.extend({"query_id": report["query_id"], "baseline_status": status, **x, "exact_f1": report["f1"] if count == len(assoc) and status == "ok" else None} for count, x in enumerate(scenarios))

    aggregate = []
    for count in range(7):
        rows = []
        for q in per_query:
            scenarios = q["retrieval_counterfactual"]
            rows.append(scenarios[min(count, len(scenarios) - 1)])
        aggregate.append({
            "assoc_count": count,
            "mean_retrieval_gold_recall": round(sum(x["gold_recall"] for x in rows) / len(rows), 4),
            "queries_with_gold": sum(x["gold_hits"] > 0 for x in rows),
            "mean_unique_papers": round(sum(x["unique_papers"] for x in rows) / len(rows), 2),
            "estimated_api_calls": round(sum(x["estimated_raw_api_calls"] for x in rows) / len(rows), 2),
            "mean_f1": None,
            "f1_status": "unavailable_without_counterfactual reranker outputs",
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "assoc_counterfactual_per_query.json").write_text(json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "assoc_counterfactual_curve.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
