"""S2-D1 (方向1): 多源召回探针——S2 search 能救回多少 OpenAlex 漏掉的 gold。

对 plan_cache 覆盖的失败查询（22 条，全 F1=0），取其全部子查询，
对每个子查询跑真实 S2 search（API key 已配置），结果按 arxiv_id / title
对齐 gold，判定：
  n_gold           : 该查询 gold 论文数
  in_OA            : gold 已进入 OpenAlex 召回池（recall_cache）
  found_by_s2      : gold 出现在 S2 search 结果里
  new_by_s2        : gold 被 S2 找到但 OpenAlex 未召回（真正的多源增益）

并行度 4，S2 429 由适配器内部重试兜底。结果写入 eval/runs/s2_d1_multisource_probe.json。
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from src.adapters.semantic_scholar import SemanticScholarAdapter
from src.telemetry import Telemetry
from eval.harness import _norm_title_letters

PLAN = Path("eval/runs/_pasa_plan_cache.jsonl")
RECALL = Path("eval/runs/_pasa_recall_cache.jsonl")
RUNS = Path("eval/runs")
GOLD = Path("data/benchmarks/pasa/RealScholarQuery/test.jsonl")
OUT = Path("eval/runs/s2_d1_multisource_probe.json")
CONCURRENCY = 4
S2_LIMIT = 20


def norm_arxiv(a: str) -> str:
    a = (a or "").strip().lower()
    m = re.match(r"(arxiv:)?([\d.]+\d+)(?:v\d+)?$", a)
    return m.group(2) if m else a


def load_plan() -> dict[str, dict]:
    out = {}
    for line in PLAN.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        subs = []
        for s in d.get("subs", []):
            t = (s.get("query_text") if isinstance(s, dict) else s) or ""
            if t:
                subs.append(t)
        out[d.get("query", "")] = {"subs": subs, "intents": [
            s.get("intent") if isinstance(s, dict) else None for s in d.get("subs", [])
        ]}
    return out


def load_recall() -> dict[str, list[dict]]:
    out = {}
    for line in RECALL.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.setdefault(d["q"], []).extend(d["evs"])
    return out


def load_gold() -> dict[str, list[dict]]:
    out = {}
    for line in GOLD.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gold = []
        aids = d.get("answer_arxiv_id") or []
        for j, a in enumerate(d.get("answer") or []):
            item = {"title": a}
            if j < len(aids) and aids[j]:
                item["arxiv_id"] = norm_arxiv(aids[j])
            gold.append(item)
        out[str(d.get("qid"))] = gold
    return out


def qid_to_query() -> dict[str, str]:
    out = {}
    for f in sorted(RUNS.glob("PASA_FULL_RealScholarQuery_*.json")):
        d = json.load(f.open(encoding="utf-8"))
        out[d["query_id"]] = d["raw_query"]
    return out


def ev_arxiv(ev: dict) -> str:
    src = ev.get("identity", {}).get("source_ids") or {}
    a = src.get("arxiv") or ""
    return norm_arxiv(a) if a else ""


def gold_in_evs(gold: dict, evs: list[dict]) -> bool:
    """gold 是否命中候选池（arxiv_id 或 title 归一化任一即算）。"""
    if gold.get("arxiv_id"):
        for ev in evs:
            if ev_arxiv(ev) == gold["arxiv_id"]:
                return True
    gn = _norm_title_letters(gold["title"])
    for ev in evs:
        if _norm_title_letters(ev["identity"].get("title") or "") == gn:
            return True
    return False


async def main() -> None:
    plan = load_plan()
    recall = load_recall()
    gold = load_gold()
    qmap = qid_to_query()

    # 反查：query_text -> 主查询（该子查询属于谁）
    sub_to_main = {}
    for mq, p in plan.items():
        for sq in p["subs"]:
            sub_to_main.setdefault(sq, []).append(mq)

    s2 = SemanticScholarAdapter(api_key=settings.semantic_scholar_api_key)
    sem = asyncio.Semaphore(CONCURRENCY)

    # 待测子查询（只测 plan_cache 覆盖的失败查询）
    all_subs: list[tuple[str, str]] = []  # (main_query, subquery)
    for mq, p in plan.items():
        for sq in p["subs"]:
            all_subs.append((mq, sq))
    print(f"待测子查询: {len(all_subs)} 条（覆盖 {len(plan)} 条主查询）")

    async def probe(mq: str, sq: str) -> tuple[str, str, list[dict]]:
        async with sem:
            evs = await s2.search(sq, limit=S2_LIMIT, telemetry=Telemetry())
            return mq, sq, [ev.model_dump() for ev in evs]

    results = await asyncio.gather(*(probe(mq, sq) for mq, sq in all_subs))

    # 汇总：主查询 -> S2 结果
    s2_by_main: dict[str, list[dict]] = {}
    for mq, sq, evs in results:
        s2_by_main.setdefault(mq, []).extend(evs)

    # 判定
    stat = Counter()
    per_query = []
    for qid, mq in qmap.items():
        p = plan.get(mq)
        if not p:
            continue
        oa_evs = []
        for sq in p["subs"]:
            oa_evs.extend(recall.get(sq, []))
        s2_evs = s2_by_main.get(mq, [])
        g_list = gold.get(qid, [])
        if not g_list:
            continue
        qstat = {"qid": qid, "n_gold": len(g_list), "in_OA": 0, "found_by_s2": 0, "new_by_s2": 0}
        for g in g_list:
            if gold_in_evs(g, oa_evs):
                qstat["in_OA"] += 1
            if gold_in_evs(g, s2_evs):
                qstat["found_by_s2"] += 1
                if not gold_in_evs(g, oa_evs):
                    qstat["new_by_s2"] += 1
        per_query.append(qstat)
        for k in ["n_gold", "in_OA", "found_by_s2", "new_by_s2"]:
            stat[k] += qstat[k]

    print("=" * 72)
    print("方向1 多源召回探针：S2 search 对失败查询的新增 gold 覆盖")
    print("=" * 72)
    print(f"  覆盖主查询: {len(per_query)} 条（全部 F1=0 失败查询）")
    print(f"  gold 论文:  {stat['n_gold']}")
    print(f"  OpenAlex 已召回: {stat['in_OA']} ({stat['in_OA']/stat['n_gold']*100:.1f}%)")
    print(f"  S2 也找到:  {stat['found_by_s2']} ({stat['found_by_s2']/stat['n_gold']*100:.1f}%)")
    print(f"  S2 新增救回 (OA未召回): {stat['new_by_s2']} "
          f"({stat['new_by_s2']/stat['n_gold']*100:.1f}%)")
    rescued_q = [q for q in per_query if q["new_by_s2"] > 0]
    print(f"\n  受影响查询: {len(rescued_q)} 条")
    for q in sorted(rescued_q, key=lambda x: -x["new_by_s2"]):
        print(f"    {q['qid']}: +{q['new_by_s2']} 篇 (S2={q['found_by_s2']}, OA={q['in_OA']}, gold={q['n_gold']})")

    out = {
        "direction": "S2-D1 multisource probe",
        "s2_limit": S2_LIMIT,
        "n_queries": len(per_query),
        "n_subqueries": len(all_subs),
        "stat": dict(stat),
        "per_query": per_query,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
