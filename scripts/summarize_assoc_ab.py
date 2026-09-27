"""Summarize comparable association-term benchmark reports."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def summarize(paths: list[str]) -> dict:
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not rows:
        raise ValueError("no reports")
    n = len(rows)
    return {
        "queries": n,
        "mean_f1": round(sum(row["f1"] for row in rows) / n, 4),
        "mean_precision": round(sum(row["precision"] for row in rows) / n, 4),
        "mean_recall": round(sum(row["recall"] for row in rows) / n, 4),
        "f1_positive": sum(row["f1"] > 0 for row in rows),
        "f1_ge_01": sum(row["f1"] >= 0.1 for row in rows),
        "f1_ge_02": sum(row["f1"] >= 0.2 for row in rows),
        "total_api_calls": sum(row["api_calls"] for row in rows),
        "mean_api_calls": round(sum(row["api_calls"] for row in rows) / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glob", required=True)
    parser.add_argument("--candidate-glob", required=True)
    args = parser.parse_args()
    baseline_paths = sorted(glob.glob(args.baseline_glob))
    candidate_paths = sorted(glob.glob(args.candidate_glob))
    baseline = summarize(baseline_paths)
    candidate = summarize(candidate_paths)
    if baseline["queries"] != candidate["queries"]:
        raise SystemExit("baseline and candidate report counts differ")
    print(json.dumps({"baseline": baseline, "candidate": candidate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
