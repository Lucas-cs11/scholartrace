"""S2-D1c (v2): 精确召回→输出丢失归因。

修正 v1 缺陷：v1 把全部查询的子查询召回合并，高估了召回覆盖。
v2 通过 plan_cache（主查询->subs）只取每条主查询自身的子查询召回集，
精确重建：召回候选 → 词法排序 → 精排池(40) → top-20 输出。

gold 论文丢失归因：
  MISS_NOT_RECALLED   : 不在该查询的任何子查询召回集中（真召回失败）
  A_lexical_excluded  : 已召回，但词法排名 >40 未进精排池
  B_reranker_downgrade: 在精排池(≤40)但未进最终 top-20 输出
  HIT_OUTPUT          : 已输出（命中）

纯离线，零 API。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import _norm_title_letters
from src.search import B3_LEX_PREKEEP

PLAN = Path("eval/runs/_pasa_plan_cache.jsonl")
RECALL = Path("eval/runs/_pasa_recall_cache.jsonl")
RUNS = Path("eval/runs")
GOLD = Path("data/benchmarks/pasa/RealScholarQuery/test.jsonl")


def tokenize(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (q or "").lower()))


def lexical_score(query: str, ident: dict) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    tt = set(tokenize(ident.get("title") or ""))
    at = set(tokenize(ident.get("abstract") or ""))
    return 3 * len(q & tt) / len(q) + len(q & at) / len(q)


def load_plan() -> dict[str, dict]:
    out = {}
    for line in PLAN.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d.get("query", "")] = d
    return out


def load_recall() -> dict[str, list[dict]]:
    out = defaultdict(list)
    for line in RECALL.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for ev in d["evs"]:
            out[d["q"]].append(ev)
    return dict(out)


def load_gold() -> dict[str, list[dict]]:
    out = {}
    for line in GOLD.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        gold = []
        for j, a in enumerate(d.get("answer") or []):
            item = {"title": a}
            aids = d.get("answer_arxiv_id") or []
            if j < len(aids) and aids[j]:
                item["arxiv_id"] = aids[j]
            gold.append(item)
        out[str(d.get("qid"))] = gold
    return out


def main() -> None:
    plan = load_plan()
    recall = load_recall()
    gold = load_gold()

    stat = Counter()
    detail = []
    per_query = []

    for f in sorted(RUNS.glob("PASA_FULL_RealScholarQuery_*.json")):
        d = json.load(f.open(encoding="utf-8"))
        qid = d["query_id"]
        mq = d["raw_query"]
        out_pids = set(d["predicted_ids"])

        # 该查询的子查询（plan_cache 里的 subs）
        p = plan.get(mq)
        if not p:
            continue
        subs = p.get("subs", [])
        # subs 可能是 [{'text':...}] 或 [str]，兼容处理
        sub_queries = []
        for s in subs:
            if isinstance(s, dict):
                t = s.get("query_text") or s.get("text") or s.get("query") or s.get("sub_query")
                if t:
                    sub_queries.append(t)
            elif isinstance(s, str):
                sub_queries.append(s)
        # 若 subs 结构不明，退化：无子查询 → 只主查询本身
        if not sub_queries:
            sub_queries = [mq]

        # 该查询召回候选（只取这些子查询的召回）
        ev_by_pid = {}
        for sq in sub_queries:
            for ev in recall.get(sq, []):
                pid = ev["identity"].get("paper_id")
                if pid:
                    ev_by_pid[pid] = ev

        # 词法排序
        scored = sorted(ev_by_pid.values(),
                        key=lambda ev: lexical_score(mq, ev["identity"]), reverse=True)
        pool_pids = {ev["identity"]["paper_id"] for ev in scored[:B3_LEX_PREKEEP]}

        # 对该查询每篇 gold 归因
        qstat = Counter()
        for g in gold.get(qid, []):
            gn = _norm_title_letters(g["title"])
            mpids = [pid for pid, ev in ev_by_pid.items()
                     if _norm_title_letters(ev["identity"].get("title") or "") == gn]
            if not mpids:
                stat["MISS_NOT_RECALLED"] += 1
                qstat["MISS_NOT_RECALLED"] += 1
                continue
            pid = mpids[0]
            if pid in out_pids:
                stat["HIT_OUTPUT"] += 1
                qstat["HIT_OUTPUT"] += 1
                continue
            if pid not in pool_pids:
                # 名次
                rank = next(i for i, ev in enumerate(scored) if ev["identity"]["paper_id"] == pid)
                stat["A_lexical_excluded"] += 1
                qstat["A_lexical_excluded"] += 1
                detail.append({"qid": qid, "gold": g["title"][:55],
                               "stage": "A_lexical_excluded", "rank": rank})
            else:
                stat["B_reranker_downgrade"] += 1
                qstat["B_reranker_downgrade"] += 1
                detail.append({"qid": qid, "gold": g["title"][:55],
                               "stage": "B_reranker_downgrade"})
        per_query.append({"qid": qid, **dict(qstat)})

    print("=" * 72)
    print("S2-D1c 精确召回→输出丢失归因（按主查询子查询召回集重建）")
    print("=" * 72)
    total = sum(stat.values())
    for k in ["MISS_NOT_RECALLED", "A_lexical_excluded", "B_reranker_downgrade", "HIT_OUTPUT"]:
        v = stat.get(k, 0)
        print(f"  {k:22s}: {v:4d} ({v/total*100:.1f}%)")
    print(f"  {'合计':22s}: {total}")
    print(f"\n  B3_LEX_PREKEEP = {B3_LEX_PREKEEP}")

    print("\n  A_lexical_excluded 名次分布（前 20 条）:")
    ex = [x for x in detail if x["stage"] == "A_lexical_excluded"]
    for x in sorted(ex, key=lambda x: x["rank"])[:20]:
        print(f"    rank={x['rank']:4d} {x['qid']} | {x['gold']}")

    out = {
        "direction": "S2-D1c precise attribution",
        "B3_LEX_PREKEEP": B3_LEX_PREKEEP,
        "stat": dict(stat),
        "total_gold_covered": total,
        "per_query": per_query,
        "detail": detail,
    }
    op = Path("eval/runs/s2_d1_attribution_v2.json")
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {op}")


if __name__ == "__main__":
    main()
