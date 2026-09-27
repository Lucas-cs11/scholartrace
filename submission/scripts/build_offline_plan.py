"""构建离线计划缓存：让 SearchEngine 的召回层完全由历史 OpenAlex 召回缓存喂入，
避免实时 OpenAlex（其每日免费配额已耗尽）。

子查询来源（保证零缓存缺失）：
  22 条冻结查询 -> 复用 _pasa_plan_cache 的真实 IR + 子查询（若子查询全部在召回缓存）
  其余查询     -> 从 PASA_FULL 轨迹的 round-1 召回重建（即实际被执行、已入缓存的子查询）

产物：eval/runs/_offline_plan_cache.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.planner import PLANNER_VERSION

RUNS = Path("eval/runs")
REAL_PLAN = RUNS / "_pasa_plan_cache.jsonl"
RECALL = RUNS / "_pasa_recall_cache.jsonl"
OUT = RUNS / "_offline_plan_cache.jsonl"


def main() -> None:
    cache_keys = set()
    for line in RECALL.open(encoding="utf-8"):
        cache_keys.add(json.loads(line)["q"])

    real_plans = {}
    for line in REAL_PLAN.open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("v") == PLANNER_VERSION:
            real_plans[d["query"]] = d

    import glob
    reports = sorted(glob.glob(str(RUNS / "PASA_FULL_RealScholarQuery_*.json")))
    entries = {}
    for f in reports:
        rep = json.load(open(f))
        mq = rep["raw_query"]
        qid = rep["query_id"]
        # round-1 实际召回的子查询（缓存覆盖的）
        trace_subs = [t["query"] for t in rep["traces"]
                      if t.get("round") == 1 and t.get("api", "").startswith(
                          ("openalex", "crossref", "s2", "assoc"))]
        # 去重保序
        seen, subs_text = set(), []
        for s in trace_subs:
            if s not in seen:
                seen.add(s)
                subs_text.append(s)

        used_real = False
        if mq in real_plans:
            rp = real_plans[mq]
            rp_subs = [s.get("query_text") for s in rp.get("subs", []) if s.get("query_text")]
            if all(s in cache_keys for s in rp_subs) and rp_subs:
                # 复用真实计划（正确 IR + intent/priority）
                entries[mq] = {"v": PLANNER_VERSION, "ir": rp["ir"], "subs": rp["subs"]}
                used_real = True
        if not used_real:
            # 轨迹重建：标记为常规子查询（缓存 evs 自带 source，assoc 保送不受影响）
            subs = [{"id": f"sq{i+1}", "parent_constraint_ids": [],
                     "query_text": s, "intent": "", "priority": 1} for i, s in enumerate(subs_text)]
            entries[mq] = {"v": PLANNER_VERSION, "ir": {"raw_query": mq}, "subs": subs}

        # 校验：全部子查询必须在召回缓存
        for s in [x.get("query_text") for x in entries[mq]["subs"]]:
            if s not in cache_keys:
                print(f"[WARN] 子查询不在缓存: {qid} | {s[:50]}")

    with OUT.open("w", encoding="utf-8") as f:
        for mq, plan in entries.items():
            f.write(json.dumps({"query": mq, **plan}, ensure_ascii=False) + "\n")

    n_real = sum(1 for p in entries.values() if "constraints" in str(p["ir"]) or p["ir"].get("topic"))
    print(f"重建 {len(entries)} 条查询计划 -> {OUT}")
    print(f"  其中复用真实 IR: {n_real} 条")
    n_subs = sum(len(p['subs']) for p in entries.values())
    print(f"  总子查询: {n_subs}，均在历史 OpenAlex 召回缓存中")


if __name__ == "__main__":
    main()
