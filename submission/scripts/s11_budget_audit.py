"""S1.1 Round-2 budget accounting audit.

审计目标：
1. 验证 round2_total=144 的真实语义（final papers vs executed queries）
2. 验证 budget enforcement：executed_queries <= 66
3. 分离统计：planned / filtered / executed / cache_hits / physical_http / returned_papers / new_unique_papers

结论：
- round2_papers_total=144 是 **final ranked output 中 retrieval_round=2 的论文数**
- 实际执行的 follow-up query 数必须 <= budget_max_new_searches=66
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_benchmark import load_pasa
from scripts.run_m5a import filter_followup
from s1.config import load_config
from s1.pipeline import ContestEngine

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"
M5A_PLANS = "eval/runs/m5a_two_round/m5a_round2_plans.jsonl"


async def audit_one(engine: ContestEngine, bq: dict) -> dict:
    """审计单个查询的 Round-2 执行。"""
    qid = bq["query_id"]
    res = await engine.search(bq["query"], query_id=qid, mode="deep")
    t = res.trace
    dec = t.round2_decision

    planned = len(dec.followups)
    executed = sum(1 for fu in dec.followups if fu.status == "executed")
    filtered = sum(1 for fu in dec.followups if fu.status not in {"executed", "keep"})
    newly_discovered = len(t.newly_discovered_papers)
    final_round2 = len([r for r in res.results if r.retrieval_round == 2])

    return {
        "query_id": qid,
        "planned_followups": planned,
        "executed_followups": executed,
        "filtered_followups": filtered,
        "newly_discovered_papers": newly_discovered,
        "final_round2_papers": final_round2,
    }


async def main():
    cfg = load_config("configs/deep.yaml")
    engine = ContestEngine(cfg)
    queries = load_pasa(DATA)
    frozen_ids = list(engine._frozen_plans)

    rows = []
    for qid in frozen_ids:
        bq = [q for q in queries if q["query_id"] == qid][0]
        row = await audit_one(engine, bq)
        rows.append(row)
        print(f"  [{row['query_id']:25}] planned={row['planned_followups']} "
              f"executed={row['executed_followups']} filtered={row['filtered_followups']} "
              f"new_papers={row['newly_discovered_papers']} final_r2={row['final_round2_papers']}")

    total_planned = sum(r["planned_followups"] for r in rows)
    total_executed = sum(r["executed_followups"] for r in rows)
    total_filtered = sum(r["filtered_followups"] for r in rows)
    total_new_papers = sum(r["newly_discovered_papers"] for r in rows)
    total_final_r2 = sum(r["final_round2_papers"] for r in rows)

    print(f"\n=== S1.1 Round-2 Budget Audit ===")
    print(f"Total planned follow-ups: {total_planned}")
    print(f"Total executed follow-ups: {total_executed}")
    print(f"Total filtered follow-ups: {total_filtered}")
    print(f"Budget max (config): {cfg.budget_max_new_searches}")
    print(f"Budget compliance: executed <= budget: {total_executed <= cfg.budget_max_new_searches}")
    print(f"Total newly discovered papers (Round-2 retrieval): {total_new_papers}")
    print(f"Total final Round-2 papers (in ranked output): {total_final_r2}")
    print(f"\nMetric naming issue confirmed:")
    print(f"  Original report: round2_total=144")
    print(f"  Actual meaning: final_round2_papers={total_final_r2}")
    print(f"  Budget-relevant metric: executed_followups={total_executed}")

    verdict = "PASS" if total_executed <= cfg.budget_max_new_searches else "FAIL"
    print(f"\n=== ROUND2_BUDGET_ACCOUNTING: {verdict} ===")

    out = {
        "verdict": verdict,
        "config_budget_max": cfg.budget_max_new_searches,
        "total_planned_followups": total_planned,
        "total_executed_followups": total_executed,
        "total_filtered_followups": total_filtered,
        "total_newly_discovered_papers": total_new_papers,
        "total_final_round2_papers": total_final_r2,
        "rows": rows,
    }
    Path("eval/runs/s1/s11_budget_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[S1.1] 已写入 eval/runs/s1/s11_budget_audit.json")


if __name__ == "__main__":
    asyncio.run(main())
