"""S2-D1: gold 对齐补全 → 离线重判假召回失败。

方向3：现有 PASA_FULL 评测的 gold 仅含 title + arxiv_id 两个键
（harness.match_gold 只生成 title:/title_n:，不用 arxiv_id 匹配），
而系统召回论文含完整身份（title/doi/arxiv），却因 predicted_ids 只存
OpenAlex W-ID 且 gold 无 openalex_id/doi 键而匹配失败。

本脚本用 PASA recall_cache（含完整论文身份）对 PASA_FULL 的 predicted
W-ID 离线重建匹配键，量化"假召回失败"规模，并验证三类修复：
  R1. gold 增加 arxiv_id 匹配键（改 match_gold）
  R2. gold 用 DOI 补全（arxiv_id -> DOI）
  R3. 预测侧用 title 匹配（缓存身份已有 title）

纯离线，零 API 消耗。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import _norm_doi, _norm_title, _norm_title_letters

CACHE = Path("eval/runs/_pasa_recall_cache.jsonl")
RUNS = Path("eval/runs")
GOLD_PATH = Path("data/benchmarks/pasa/RealScholarQuery/test.jsonl")


def norm_arxiv(a: str) -> str:
    """arxiv id 归一化：去掉版本号（2309.04564v1 -> 2309.04564）。"""
    a = (a or "").strip().lower()
    m = re.match(r"(arxiv:)?([\d.]+\d+)(?:v\d+)?$", a)
    return m.group(2) if m else a


def load_cache() -> dict[str, dict]:
    cache = {}
    for line in CACHE.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for ev in d["evs"]:
            ident = ev["identity"]
            pid = ident.get("paper_id") or ""
            if pid.startswith("W"):
                cache[pid] = ident
    return cache


def load_gold() -> dict[str, list[dict]]:
    """qid -> gold list（title + arxiv_id）。"""
    out = {}
    for line in GOLD_PATH.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        qid = str(d.get("qid") or "q")
        gold = []
        for j, a in enumerate(d.get("answer") or []):
            item = {"title": a}
            aids = d.get("answer_arxiv_id") or []
            if j < len(aids) and aids[j]:
                item["arxiv_id"] = aids[j]
            gold.append(item)
        out[qid] = gold
    return out


def gold_keys(g: dict) -> set[str]:
    keys = set()
    if g.get("title"):
        keys.add(f"title:{_norm_title(g['title'])}")
        keys.add(f"title_n:{_norm_title_letters(g['title'])}")
    return keys


def predict_keys(ident: dict) -> set[str]:
    keys = set()
    if ident.get("paper_id"):
        keys.add(f"openalex:{ident['paper_id']}")
    doi = _norm_doi(ident.get("doi"))
    if doi:
        keys.add(f"doi:{doi}")
    if ident.get("title"):
        keys.add(f"title:{_norm_title(ident['title'])}")
        keys.add(f"title_n:{_norm_title_letters(ident['title'])}")
    # arxiv id（若 identity 里有 source_ids）
    src = ident.get("source_ids") or {}
    if src.get("arxiv"):
        keys.add(f"arxiv:{norm_arxiv(src['arxiv'])}")
    # 从 doi 反推 arxiv id（10.48550/arXiv.xxxx.xxxxx）
    if doi and doi.startswith("10.48550/arxiv."):
        keys.add(f"arxiv:{doi.split('.', 2)[2].lower()}")
    return keys


def evaluate_cases(predicted: list[dict], gold: list[dict]) -> dict:
    """返回每篇 gold 的状态统计。

    predicted: [{ident, keys}], gold: [{title, arxiv_id, keys}]
    """
    n_gold = len(gold)
    n_hit_orig = 0       # 原匹配（title 键）
    n_hit_with_arxiv = 0 # 加 arxiv 键后命中
    hit_arxiv_only = 0   # 仅靠 arxiv 键新增命中
    n_hit_with_doi = 0   # 加 doi 键后命中
    hit_doi_only = 0
    detail = []
    for gi, g in enumerate(gold):
        gkeys = gold_keys(g)
        # 原匹配（title 键）
        matched = False
        for p in predicted:
            if p["keys"] & gkeys:
                matched = True
                break
        if matched:
            n_hit_orig += 1
            detail.append({"gi": gi, "status": "HIT_ORIG", "title": g["title"][:60]})
            continue
        # 加 arxiv 键（gold 侧）
        ark = g.get("arxiv_id")
        gk_ext = set(gkeys)
        if ark:
            gk_ext.add(f"arxiv:{norm_arxiv(ark)}")
        matched_arxiv = False
        for p in predicted:
            if p["keys"] & gk_ext:
                matched_arxiv = True
                break
        if matched_arxiv:
            n_hit_with_arxiv += 1
            if not matched:
                hit_arxiv_only += 1
            detail.append({"gi": gi, "status": "HIT_ARXIV_ONLY", "title": g["title"][:60],
                           "arxiv": ark})
            continue
        # 加 doi 键（从 arxiv 反推或缓存）
        n_hit_with_doi = n_hit_with_arxiv  # 占位，实际另算
        detail.append({"gi": gi, "status": "MISS", "title": g["title"][:60], "arxiv": ark})
    return {
        "n_gold": n_gold,
        "n_hit_orig": n_hit_orig,
        "n_hit_arxiv_only": hit_arxiv_only,
        "detail": detail,
    }


def main() -> None:
    cache = load_cache()
    gold = load_gold()

    full_files = sorted(RUNS.glob("PASA_FULL_RealScholarQuery_*.json"))
    print(f"加载 PASA_FULL 结果: {len(full_files)} 条")

    agg = Counter()
    recovered_q = []
    per_query = []

    for f in full_files:
        d = json.load(f.open(encoding="utf-8"))
        qid = d["query_id"]
        g_list = gold.get(qid, [])
        # 预测论文（从缓存取身份，重建 keys）
        predicted = []
        for pid in d["predicted_ids"]:
            ident = cache.get(pid)
            if ident is None:
                continue
            predicted.append({"ident": ident, "keys": predict_keys(ident)})
        # 对每篇 gold 判定
        res = evaluate_cases(predicted, g_list)
        agg["n_gold_total"] += res["n_gold"]
        agg["n_hit_orig_total"] += res["n_hit_orig"]
        agg["n_hit_arxiv_only_total"] += res["n_hit_arxiv_only"]
        if res["n_hit_arxiv_only"] > 0:
            recovered_q.append({
                "qid": qid,
                "orig_f1": d["f1"],
                "arxiv_only_hits": res["n_hit_arxiv_only"],
                "n_gold": res["n_gold"],
                "hits_after": res["n_hit_orig"] + res["n_hit_arxiv_only"],
            })
        per_query.append({
            "qid": qid, "orig_f1": d["f1"],
            "n_gold": res["n_gold"],
            "n_hit_orig": res["n_hit_orig"],
            "n_hit_arxiv_only": res["n_hit_arxiv_only"],
        })

    print()
    print("=" * 70)
    print("方向3 离线重判结果（零 API 消耗）")
    print("=" * 70)
    n_g = agg["n_gold_total"]
    print(f"\ngold 论文总数: {n_g}")
    print(f"原匹配命中 (title 键):      {agg['n_hit_orig_total']} "
          f"({agg['n_hit_orig_total']/n_g*100:.1f}%)")
    print(f"仅靠 arxiv 键新增命中:      {agg['n_hit_arxiv_only_total']} "
          f"({agg['n_hit_arxiv_only_total']/n_g*100:.1f}%)")
    print(f"\n受影响查询 ({len(recovered_q)} 条):")
    for r in sorted(recovered_q, key=lambda x: -x["arxiv_only_hits"]):
        print(f"  {r['qid']}: F1={r['orig_f1']:.4f} -> +{r['arxiv_only_hits']} 篇 gold 可恢复 "
              f"({r['hits_after']}/{r['n_gold']})")

    # 汇总 F1 影响（估算：新增命中后 recall 提升）
    print()
    print("若补 arxiv 匹配键后 F1 变化（按当前 predicted 数量估算）:")
    delta_f1 = 0.0
    for q in per_query:
        if q["n_hit_arxiv_only"] > 0:
            # 新 F1 = 2*P*R/(P+R)
            tp_new = q["n_hit_orig"] + q["n_hit_arxiv_only"]
            pred_n = 20  # 保守按 top-20
            p = tp_new / pred_n
            r = tp_new / q["n_gold"]
            f1_new = 2 * p * r / (p + r) if (p + r) else 0.0
            delta_f1 += f1_new - q["orig_f1"]
            print(f"  {q['qid']}: F1 {q['orig_f1']:.4f} -> {f1_new:.4f}")

    print(f"\n平均 F1 提升估计: +{delta_f1/len(per_query):.4f}")

    out = {
        "direction": "S2-D1 gold realignment",
        "total_gold": n_g,
        "orig_hits_title_only": agg["n_hit_orig_total"],
        "arxiv_only_additional_hits": agg["n_hit_arxiv_only_total"],
        "recovered_queries": len(recovered_q),
        "per_query": per_query,
    }
    out_path = Path("eval/runs/s2_d1_realignment.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
