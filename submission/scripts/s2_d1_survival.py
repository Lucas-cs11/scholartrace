"""方向1 确定性生存分析：S2 新增救回的 gold 有多少能进精排池。

用 run2 (PASA_S2MS_NOCIT) 的召回缓存（含 OpenAlex + S2 双路），重建每条
冻结查询的组合候选池 → 词法排序 → 精排池(B3_LEX_PREKEEP=40)，判定：

  gold_in_pool           : gold 进入精排池（召回 + 词法≤40）
  gold_recalled_but_excl : gold 被召回但词法排名>40 被挤出（S2 招回但词法筛掉）
  new_by_s2_in_pool      : 仅被 S2 召回、OpenAlex 未召回，且进入精排池（真正多源增益）
  new_by_s2_excluded     : 仅被 S2 召回但被词法挤出精排池

纯离线，零 API。结果写入 eval/runs/s2_d1_survival.json。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search import B3_LEX_PREKEEP
from eval.harness import _norm_title_letters

PLAN = Path("eval/runs/_s2ms_plan_cache.jsonl")
RECALL = Path("eval/runs/_s2ms_recall_cache.jsonl")  # run2 缓存
RUNS = Path("eval/runs")
GOLD = Path("data/benchmarks/pasa/RealScholarQuery/test.jsonl")
OUT = Path("eval/runs/s2_d1_survival.json")


def tokenize(q: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (q or "").lower()))


def lex_score(query: str, ident: dict) -> float:
    q = tokenize(query)
    if not q:
        return 0.0
    tt = set(tokenize(ident.get("title") or ""))
    at = set(tokenize(ident.get("abstract") or ""))
    return (len(q & tt) * 3.0 + len(q & at)) / (len(q) * 3.0 + 1e-9)


def norm_arxiv(a: str) -> str:
    a = (a or "").strip().lower()
    m = re.match(r"(arxiv:)?([\d.]+\d+)(?:v\d+)?$", a)
    return m.group(2) if m else a


def load_plan() -> dict[str, list[str]]:
    out = {}
    for line in PLAN.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out[d.get("query", "")] = [
            s.get("query_text") for s in d.get("subs", []) if s.get("query_text")
        ]
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
    return norm_arxiv(src.get("arxiv") or "")


def gold_matches_ev(g: dict, ev: dict) -> bool:
    if g.get("arxiv_id") and ev_arxiv(ev) == g["arxiv_id"]:
        return True
    return _norm_title_letters(ev["identity"].get("title") or "") == _norm_title_letters(g["title"])


def main() -> None:
    plan = load_plan()
    recall = load_recall()
    gold = load_gold()
    qmap = qid_to_query()

    stat = Counter()
    per_query = []

    for qid, mq in qmap.items():
        subs = plan.get(mq, [mq])
        oa_evs, s2_evs, all_evs = [], [], []
        for sq in subs:
            for ev in recall.get(sq, []):
                all_evs.append(ev)
                if ev.get("source") == "s2_recall":
                    s2_evs.append(ev)
                else:
                    oa_evs.append(ev)
        # 去重按 paper_id
        def dedup(evs):
            out, seen = [], set()
            for ev in evs:
                pid = ev["identity"].get("paper_id")
                if pid and pid not in seen:
                    seen.add(pid)
                    out.append(ev)
            return out
        all_evs = dedup(all_evs)
        oa_evs = dedup(oa_evs)
        # 词法排序
        scored = sorted(all_evs, key=lambda ev: lex_score(mq, ev["identity"]), reverse=True)
        pool_pids = {ev["identity"]["paper_id"] for ev in scored[:B3_LEX_PREKEEP]}

        qstat = {"qid": qid, "gold_in_pool": 0, "gold_excluded": 0,
                 "new_by_s2_in_pool": 0, "new_by_s2_excluded": 0}
        for g in gold.get(qid, []):
            in_oa = any(gold_matches_ev(g, ev) for ev in oa_evs)
            in_s2 = any(gold_matches_ev(g, ev) for ev in s2_evs)
            in_pool = any(gold_matches_ev(g, ev) for ev in scored[:B3_LEX_PREKEEP])
            if in_pool:
                stat["gold_in_pool"] += 1
                qstat["gold_in_pool"] += 1
                if in_s2 and not in_oa:
                    stat["new_by_s2_in_pool"] += 1
                    qstat["new_by_s2_in_pool"] += 1
            else:
                stat["gold_excluded"] += 1
                qstat["gold_excluded"] += 1
                if in_s2 and not in_oa:
                    stat["new_by_s2_excluded"] += 1
                    qstat["new_by_s2_excluded"] += 1
        per_query.append(qstat)

    total_gold = sum(q["gold_in_pool"] + q["gold_excluded"] for q in per_query)
    print("=" * 72)
    print("方向1 确定性生存分析（22 冻结查询，S2+OpenAlex 组合池）")
    print("=" * 72)
    print(f"  gold 论文: {total_gold}")
    print(f"  进入精排池 (词法≤{B3_LEX_PREKEEP}): {stat['gold_in_pool']} "
          f"({stat['gold_in_pool']/total_gold*100:.1f}%)")
    print(f"  被词法挤出: {stat['gold_excluded']} "
          f"({stat['gold_excluded']/total_gold*100:.1f}%)")
    print(f"  ├─ 其中 S2 新增救回且进池: {stat['new_by_s2_in_pool']} "
          f"({stat['new_by_s2_in_pool']/total_gold*100:.1f}%)")
    print(f"  ├─ 其中 S2 新增救回但被挤出: {stat['new_by_s2_excluded']} "
          f"({stat['new_by_s2_excluded']/total_gold*100:.1f}%)")

    out = {"direction": "S2-D1 survival", "B3_LEX_PREKEEP": B3_LEX_PREKEEP,
           "stat": dict(stat), "total_gold": total_gold, "per_query": per_query}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
