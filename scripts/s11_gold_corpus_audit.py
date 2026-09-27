"""S1.1 Gold corpus version audit.

审计目标：
1. 确认当前 evaluator Gold 总数（180 vs 184）
2. 追溯数据源和去重口径
3. 输出 corpus fingerprint 供后续引用
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.eval_benchmark import load_pasa  # noqa: E402

DATA = "data/benchmarks/pasa/RealScholarQuery/test.jsonl"


def main():
    queries = load_pasa(DATA)

    # 按 harness match_gold 口径统计（paper-level unique）
    total_query_gold_count = 0
    total_gold_instances = 0
    gold_identity_keys = set()

    for q in queries:
        gold = q.get("gold", [])
        total_gold_instances += len(gold)
        for g in gold:
            # match_gold 逻辑：每篇 gold 论文生成一组 identity keys（openalex_id/doi/title/title_n）
            keys = []
            if g.get("openalex_id"):
                keys.append(f"openalex:{g['openalex_id']}")
            doi = g.get("doi") or g.get("openalex_doi")
            if doi:
                # 归一化 DOI（eval.harness._norm_doi）
                doi_norm = doi.lower().strip()
                if doi_norm.startswith("https://doi.org/"):
                    doi_norm = doi_norm.replace("https://doi.org/", "")
                keys.append(f"doi:{doi_norm}")
            if g.get("title"):
                # eval.harness._norm_title / _norm_title_letters
                title_norm = " ".join(g["title"].lower().split())
                title_n = "".join(c for c in g["title"].lower() if c.isalnum())
                keys.append(f"title:{title_norm}")
                keys.append(f"title_n:{title_n}")

            # 每组 keys 代表一篇 paper-level Gold
            gold_identity_keys.update(keys)

        total_query_gold_count += len(gold)

    # Corpus fingerprint
    sorted_keys = sorted(gold_identity_keys)
    corpus_hash = hashlib.sha256("".join(sorted_keys).encode()).hexdigest()[:16]

    # Data source fingerprint
    data_content = Path(DATA).read_bytes()
    data_hash = hashlib.sha256(data_content).hexdigest()[:16]

    print(f"=== S1.1 Gold Corpus Version Audit ===")
    print(f"Data source: {DATA}")
    print(f"Data source hash: {data_hash}")
    print(f"Total queries: {len(queries)}")
    print(f"Total query-gold instances (sum of len(gold) per query): {total_query_gold_count}")
    print(f"Total unique Gold identity keys (openalex_id/doi/title/title_n): {len(gold_identity_keys)}")
    print(f"Corpus fingerprint (sha256[:16] of sorted keys): {corpus_hash}")

    # 检查 evaluator 实际计算的 total_gold_n
    # S1 eval 报告的 184 是 sum(len(match_gold(q))) over all queries
    from eval.harness import match_gold
    evaluator_gold_count = sum(len(match_gold(q)) for q in queries)

    print(f"\n=== Evaluator Gold Count ===")
    print(f"Evaluator total_gold_papers (sum of len(match_gold(q))): {evaluator_gold_count}")
    print(f"  (match_gold 返回每篇 Gold 的 identity key 集合，len(match_gold(q)) = 该查询 paper-level Gold 数)")

    # 180 vs 184 分析
    print(f"\n=== 180 vs 184 Analysis ===")
    print(f"Historical reports: 180 Gold")
    print(f"S1 eval: {evaluator_gold_count} Gold")
    print(f"Discrepancy: {evaluator_gold_count - 180}")

    # 检查是否是 frozen 22-query subset vs full 50-query set
    frozen_22_ids = [
        "RealScholarQuery_0", "RealScholarQuery_6", "RealScholarQuery_8",
        "RealScholarQuery_14", "RealScholarQuery_15", "RealScholarQuery_17",
        "RealScholarQuery_20", "RealScholarQuery_21", "RealScholarQuery_22",
        "RealScholarQuery_23", "RealScholarQuery_25", "RealScholarQuery_28",
        "RealScholarQuery_29", "RealScholarQuery_34", "RealScholarQuery_35",
        "RealScholarQuery_38", "RealScholarQuery_39", "RealScholarQuery_41",
        "RealScholarQuery_42", "RealScholarQuery_43", "RealScholarQuery_47",
        "RealScholarQuery_48",
    ]
    frozen_queries = [q for q in queries if q["query_id"] in frozen_22_ids]
    frozen_gold_count = sum(len(match_gold(q)) for q in frozen_queries)

    print(f"\n=== Frozen 22-Query Subset ===")
    print(f"Frozen 22 queries Gold: {frozen_gold_count}")
    print(f"Full 50 queries Gold: {evaluator_gold_count}")

    # 假设：历史 180 可能是 challenges_v1.jsonl（20 queries）
    challenges_v1_gold = sum(len(json.loads(l).get("gold", []))
                             for l in Path("eval/gold/challenges_v1.jsonl").read_text(encoding="utf-8").splitlines()
                             if l.strip())
    print(f"\n=== challenges_v1.jsonl (20 queries) ===")
    print(f"Total Gold instances: {challenges_v1_gold}")

    report = {
        "data_source": DATA,
        "data_hash": data_hash,
        "corpus_fingerprint": corpus_hash,
        "total_queries": len(queries),
        "frozen_22_queries": len(frozen_queries),
        "query_gold_instances": total_query_gold_count,
        "unique_gold_identity_keys": len(gold_identity_keys),
        "evaluator_total_gold_papers": evaluator_gold_count,
        "frozen_22_gold_papers": frozen_gold_count,
        "challenges_v1_gold_instances": challenges_v1_gold,
        "historical_180_hypothesis": "challenges_v1.jsonl (20 queries) or different de-dup logic",
        "s1_184_confirmed": evaluator_gold_count == 184,
    }

    Path("eval/runs/s1/s11_gold_corpus_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[S1.1] 已写入 eval/runs/s1/s11_gold_corpus_audit.json")

    print(f"\n=== Conclusion ===")
    print(f"EVAL_CORPUS_VERSION = pasa_realscholar_test_{data_hash}")
    print(f"Total Gold (S1 frozen 22 queries): {frozen_gold_count}")
    print(f"Total Gold (full 50 queries): {evaluator_gold_count}")


if __name__ == "__main__":
    main()
