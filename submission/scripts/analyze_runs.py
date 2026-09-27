"""方向1/2/4/5 实验对比聚合：跨 PASA_* 实验报告，输出确定性 + 端到端指标。

区分两类信号：
  - 确定性（可靠）：召回池 gold 覆盖（来自各 run 的 recall_cache）、per-query 命中数
  - 端到端（含 LLM 精排随机性，需谨慎）：P/R/F1，来自 harness 报告

对比限定在 plan_cache 冻结的 22 条查询（全 F1=0 失败集）做严格 A/B；
另给全 50 条的整体数字（注：28 条非冻结查询计划为 run1 生成，跨 run 一致）。

用法：python3 scripts/analyze_runs.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RUNS = Path("eval/runs")
FROZEN_QIDS = set()  # plan_cache 覆盖的主查询 -> 其 qid（由 PASA_FULL 报告反查）

EXPERIMENTS = ["PASA_NOCIT", "PASA_S2MS_NOCIT", "PASA_S2CIT", "PASA_S2TLDR", "PASA_S2CIT3"]

# 22 条冻结计划的 qid（来自 _pasa_plan_cache 覆盖的主查询，经 PASA_FULL 反查）
def frozen_qids() -> set[str]:
    plans = {json.loads(l)["query"] for l in
             (RUNS / "_pasa_plan_cache.jsonl").open(encoding="utf-8")
             if l.strip()}
    out = set()
    for f in glob.glob(str(RUNS / "PASA_FULL_RealScholarQuery_*.json")):
        rep = json.load(open(f))
        if rep["raw_query"] in plans:
            out.add(rep["query_id"])
    return out


def load_reports(exp: str) -> dict[str, dict]:
    out = {}
    for f in glob.glob(str(RUNS / f"{exp}_RealScholarQuery_*.json")):
        rep = json.load(open(f))
        out[rep["query_id"]] = rep
    return out


def macro_avg(reports: dict[str, dict]) -> dict:
    """宏平均 P/R/F1（每查询等权）。"""
    if not reports:
        return {"p": 0, "r": 0, "f1": 0, "n": 0}
    p = sum(r["precision"] for r in reports.values()) / len(reports)
    r = sum(r["recall"] for r in reports.values()) / len(reports)
    f1 = sum(r["f1"] for r in reports.values()) / len(reports)
    return {"p": p, "r": r, "f1": f1, "n": len(reports)}


def main() -> None:
    frozen = frozen_qids()
    print(f"冻结计划查询数（严格A/B集）: {len(frozen)}")
    # 用 PASA_FULL 作为历史基线参考（引文走 OpenCitations，无 S2）
    baselines = {"PASA_FULL": load_reports("PASA_FULL")}

    tables = {}
    for exp in EXPERIMENTS:
        reps = load_reports(exp)
        tables[exp] = reps
        if not reps:
            continue
        all_q = macro_avg(reps)
        sub = {q: r for q, r in reps.items() if q in frozen}
        frz = macro_avg(sub)
        print(f"\n=== {exp} ===")
        print(f"  全 50 条: F1={all_q['f1']:.4f} P={all_q['p']:.4f} R={all_q['r']:.4f} (n={all_q['n']})")
        print(f"  22 冻结:  F1={frz['f1']:.4f} P={frz['p']:.4f} R={frz['r']:.4f} (n={frz['n']})")

    # 历史 PASA_FULL（仅 22 冻结可严格比）
    pf_sub = {q: r for q, r in baselines["PASA_FULL"].items() if q in frozen}
    print(f"\n=== PASA_FULL (历史: OpenAlex+OpenCitations, 22冻结) ===")
    frz = macro_avg(pf_sub)
    print(f"  22 冻结:  F1={frz['f1']:.4f} P={frz['p']:.4f} R={frz['r']:.4f}")

    # 方向1: S2MS_NOCIT vs NOCIT (两者共用 22 冻结计划，纯处理差异 = S2 召回)
    print("\n" + "=" * 72)
    print("方向1 端到端增益（22 冻结查询，S2多源召回 vs 纯OpenAlex，均无引文）")
    print("=" * 72)
    if "PASA_NOCIT" in tables and "PASA_S2MS_NOCIT" in tables:
        a = {q: r for q, r in tables["PASA_NOCIT"].items() if q in frozen}
        b = {q: r for q, r in tables["PASA_S2MS_NOCIT"].items() if q in frozen}
        ma, mb = macro_avg(a), macro_avg(b)
        print(f"  NOCIT   : F1={ma['f1']:.4f} P={ma['p']:.4f} R={ma['r']:.4f} 命中查询={sum(1 for r in a.values() if r['f1']>0)}")
        print(f"  S2MS    : F1={mb['f1']:.4f} P={mb['p']:.4f} R={mb['r']:.4f} 命中查询={sum(1 for r in b.values() if r['f1']>0)}")
        print(f"  Δ F1={mb['f1']-ma['f1']:+.4f}  Δ P={mb['p']-ma['p']:+.4f}  Δ R={mb['r']-ma['r']:+.4f}")
        # 逐查询 F1 变好/变坏/不变
        diffs = []
        for q in frozen:
            fa = a[q]["f1"] if q in a else 0
            fb = b[q]["f1"] if q in b else 0
            diffs.append((q, fa, fb))
        better = [d for d in diffs if d[2] > d[1]]
        worse = [d for d in diffs if d[2] < d[1]]
        print(f"  F1 变好 {len(better)} 条, 变坏 {len(worse)} 条, 不变 {len(diffs)-len(better)-len(worse)} 条")
        for q, fa, fb in sorted(better, key=lambda x: x[2]-x[1], reverse=True)[:8]:
            print(f"    ↑ {q}: {fa:.4f} -> {fb:.4f}")
        for q, fa, fb in sorted(worse, key=lambda x: x[1]-x[2], reverse=True)[:5]:
            print(f"    ↓ {q}: {fa:.4f} -> {fb:.4f}")

    # 方向2: S2CIT vs NOCIT（引文开启[S2] vs 关闭，均无S2召回）
    print("\n" + "=" * 72)
    print("方向2 端到端增益（22 冻结查询，S2引文扩展 开 vs 关）")
    print("=" * 72)
    if "PASA_NOCIT" in tables and "PASA_S2CIT" in tables:
        a = {q: r for q, r in tables["PASA_NOCIT"].items() if q in frozen}
        b = {q: r for q, r in tables["PASA_S2CIT"].items() if q in frozen}
        ma, mb = macro_avg(a), macro_avg(b)
        print(f"  NOCIT : F1={ma['f1']:.4f} P={ma['p']:.4f} R={ma['r']:.4f}")
        print(f"  S2CIT : F1={mb['f1']:.4f} P={mb['p']:.4f} R={mb['r']:.4f}")
        print(f"  Δ F1={mb['f1']-ma['f1']:+.4f}  Δ P={mb['p']-ma['p']:+.4f}  Δ R={mb['r']-ma['r']:+.4f}")

    # 方向4: S2TLDR vs S2MS_NOCIT（tldr 精排 vs 普通精排，均 S2 召回无引文）
    print("\n" + "=" * 72)
    print("方向4 端到端增益（22 冻结，S2召回下 tldr 精排 开 vs 关）")
    print("=" * 72)
    if "PASA_S2MS_NOCIT" in tables and "PASA_S2TLDR" in tables:
        a = {q: r for q, r in tables["PASA_S2MS_NOCIT"].items() if q in frozen}
        b = {q: r for q, r in tables["PASA_S2TLDR"].items() if q in frozen}
        ma, mb = macro_avg(a), macro_avg(b)
        print(f"  S2MS_NOCIT : F1={ma['f1']:.4f} P={ma['p']:.4f} R={ma['r']:.4f}")
        print(f"  S2TLDR     : F1={mb['f1']:.4f} P={mb['p']:.4f} R={mb['r']:.4f}")
        print(f"  Δ F1={mb['f1']-ma['f1']:+.4f}  Δ P={mb['p']-ma['p']:+.4f}  Δ R={mb['r']-ma['r']:+.4f}")

    # 方向5: S2CIT3 vs S2CIT（3跳 vs 2跳 引文扩展）
    print("\n" + "=" * 72)
    print("方向5 端到端增益（22 冻结，引文深度 3跳 vs 2跳）")
    print("=" * 72)
    if "PASA_S2CIT" in tables and "PASA_S2CIT3" in tables:
        a = {q: r for q, r in tables["PASA_S2CIT"].items() if q in frozen}
        b = {q: r for q, r in tables["PASA_S2CIT3"].items() if q in frozen}
        ma, mb = macro_avg(a), macro_avg(b)
        print(f"  2跳 : F1={ma['f1']:.4f} P={ma['p']:.4f} R={ma['r']:.4f}")
        print(f"  3跳 : F1={mb['f1']:.4f} P={mb['p']:.4f} R={mb['r']:.4f}")
        print(f"  Δ F1={mb['f1']-ma['f1']:+.4f}  Δ P={mb['p']-ma['p']:+.4f}  Δ R={mb['r']-ma['r']:+.4f}")

    # 汇总表保存
    out = {}
    for exp, reps in tables.items():
        all_q = macro_avg(reps)
        sub = macro_avg({q: r for q, r in reps.items() if q in frozen})
        out[exp] = {"all50": all_q, "frozen22": sub}
    pf_sub = macro_avg(pf_sub)
    out["PASA_FULL"] = {"all50": macro_avg(baselines["PASA_FULL"]), "frozen22": pf_sub}
    (RUNS / "s2_experiments_aggregate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {RUNS/'s2_experiments_aggregate.json'}")


if __name__ == "__main__":
    main()
