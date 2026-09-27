"""S2-D1b: 召回→输出丢失归因。

对每条 PASA_FULL 查询，用缓存重建候选池的词法排序，判断被召回但
未输出的 gold 论文丢失在哪一环：
  A. lexical_excluded   : 词法 top-B3_LEX_PREKEEP(40) 之外，未进精排池
  B. reranker_downgraded: 在精排池内但 LLM 给低分/截断，未进 top-20
  C. assoc_missing      : assoc 来源但未保送（不应发生，保送开着）

纯离线，零 API 消耗。依赖：PASA recall_cache + PASA_FULL 结果。
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

CACHE = Path("eval/runs/_pasa_recall_cache.jsonl")
RUNS = Path("eval/runs")
GOLD_PATH = Path("data/benchmarks/pasa/RealScholarQuery/test.jsonl")


def tokenize(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", query.lower()))


def lexical_score(query: str, ident: dict) -> float:
    """复刻 search.py _lexical_rank 的词法评分（title 权重3，abstract 权重1）。"""
    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0
    title_t = set(tokenize(ident.get("title") or ""))
    abs_t = set(tokenize(ident.get("abstract") or ""))
    title_overlap = len(q_tokens & title_t) / len(q_tokens)
    abs_overlap = len(q_tokens & abs_t) / len(q_tokens)
    return 3 * title_overlap + 1 * abs_overlap


def norm_arxiv(a: str) -> str:
    a = (a or "").strip().lower()
    m = re.match(r"(arxiv:)?([\d.]+\d+)(?:v\d+)?$", a)
    return m.group(2) if m else a


def load_cache() -> dict[str, list[dict]]:
    cache = defaultdict(list)
    for line in CACHE.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for ev in d["evs"]:
            cache[d["q"]].append(ev)
    return dict(cache)


def load_gold() -> dict[str, list[dict]]:
    out = {}
    for line in GOLD_PATH.open(encoding="utf-8"):
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
    cache_by_q = load_cache()
    gold = load_gold()

    # 重建 query 文本 -> 主 query 映射（从 plan_cache 取子查询对应的主查询）
    # 简化：PASA_FULL 的 raw_query 即主查询，子查询文本在 plan_cache 里
    plan_cache = {}
    pc_path = Path("eval/runs/_pasa_plan_cache.jsonl")
    if pc_path.exists():
        for line in pc_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            plan_cache[d.get("query", "")] = d

    # 主 query 文本
    qid_to_query = {}
    for f in sorted(RUNS.glob("PASA_FULL_RealScholarQuery_*.json")):
        d = json.load(f.open(encoding="utf-8"))
        qid_to_query[d["query_id"]] = d["raw_query"]

    # 统计
    stat = Counter()
    detail = []

    for qid, main_query in qid_to_query.items():
        gold_list = gold.get(qid, [])
        # 该查询所有召回论文（缓存合并去重）
        ev_by_pid = {}
        for q, evs in cache_by_q.items():
            for ev in evs:
                pid = ev["identity"].get("paper_id")
                if pid:
                    ev_by_pid[pid] = ev
        # 词法排序（用主 query 评分，近似 search_full 行为）
        scored = sorted(ev_by_pid.values(),
                        key=lambda ev: lexical_score(main_query, ev["identity"]),
                        reverse=True)
        pool_pids = {ev["identity"]["paper_id"] for ev in scored[:B3_LEX_PREKEEP]}
        out_pids = set()
        for f in sorted(RUNS.glob(f"PASA_FULL_{qid}.json")):
            d = json.load(f.open(encoding="utf-8"))
            out_pids = set(d["predicted_ids"])
        # 但 PASA_FULL 文件名是 PASA_FULL_RealScholarQuery_N.json，query_id 即 RealScholarQuery_N
        out_pids = set()
        fp = RUNS / f"PASA_FULL_{qid}.json"
        if fp.exists():
            out_pids = set(json.load(fp.open(encoding="utf-8"))["predicted_ids"])

        for g in gold_list:
            gn = _norm_title_letters(g["title"])
            # 找到缓存中匹配的论文
            matched_pids = [pid for pid, ev in ev_by_pid.items()
                            if _norm_title_letters(ev["identity"].get("title") or "") == gn]
            if not matched_pids:
                continue  # 未召回
            pid = matched_pids[0]
            if pid in out_pids:
                continue  # 已输出，命中
            # 丢失归因
            if pid not in pool_pids:
                stat["A_lexical_excluded"] += 1
                detail.append({"qid": qid, "gold": g["title"][:60], "stage": "A_lexical_excluded"})
            else:
                stat["B_reranker_downgraded"] += 1
                detail.append({"qid": qid, "gold": g["title"][:60], "stage": "B_reranker_downgraded"})

    print("=" * 70)
    print("召回→输出丢失归因（75 篇被召回但未输出的 gold）")
    print("=" * 70)
    for k, v in stat.most_common():
        print(f"  {k}: {v}")
    print(f"  合计: {sum(stat.values())}")
    print()
    print("丢失明细（前 15 条）:")
    for d in detail[:15]:
        print(f"  {d['qid']} [{d['stage']}]: {d['gold']}")

    out = {
        "direction": "S2-D1b attribution",
        "B3_LEX_PREKEEP": B3_LEX_PREKEEP,
        "attribution": dict(stat),
        "total": sum(stat.values()),
        "detail": detail,
    }
    op = Path("eval/runs/s2_d1_attribution.json")
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {op}")


if __name__ == "__main__":
    main()
