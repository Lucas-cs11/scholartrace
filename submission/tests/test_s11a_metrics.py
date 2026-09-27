"""S1.1A metric naming and schema tests (no LLM required)."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_eval_schema_no_ambiguous_raw_gold():
    """Verify run_eval.py code does not use deprecated 'raw_gold' / 'gold_n'."""
    code = Path(REPO / "scripts/run_eval.py").read_text(encoding="utf-8")

    # Allow in comments/docstrings explaining the old field
    lines = [l for l in code.split("\n") if not l.strip().startswith("#") and '"""' not in l]
    code_only = "\n".join(lines)

    assert '"raw_gold"' not in code_only, "run_eval.py must not use deprecated 'raw_gold' field"
    assert '"gold_n"' not in code_only, "run_eval.py must not use deprecated 'gold_n' field"


def test_eval_schema_uses_standardized_fields():
    """Verify run_eval.py uses standardized metric field names."""
    code = Path(REPO / "scripts/run_eval.py").read_text(encoding="utf-8")

    required = [
        "final_unique_gold",
        "total_gold_papers",
        "retrieval_total_candidates",
        "final_output_size",
        "eval_corpus_version",
    ]
    for field in required:
        assert field in code, f"run_eval.py must use standardized field '{field}'"


def test_round2_budget_configured():
    """Verify DEEP config has explicit budget limit."""
    import yaml
    with open(REPO / "configs/deep.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "budget_max_new_searches" in cfg
    assert cfg["budget_max_new_searches"] == 66


def test_eval_corpus_version_exists():
    """Verify corpus version constant is defined."""
    code = Path(REPO / "scripts/run_eval.py").read_text(encoding="utf-8")
    assert "EVAL_CORPUS_VERSION" in code
    assert "pasa_realscholar_test" in code


def test_frozen_22_gold_count():
    """Verify frozen 22-query Gold count is 184."""
    from eval.harness import match_gold
    from scripts.eval_benchmark import load_pasa

    queries = load_pasa("data/benchmarks/pasa/RealScholarQuery/test.jsonl")
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
    frozen = [q for q in queries if q["query_id"] in frozen_22_ids]
    total_gold = sum(len(match_gold(q)) for q in frozen)
    assert total_gold == 184, f"Frozen 22 queries must have 184 Gold, got {total_gold}"


def test_round2_metric_naming():
    """Verify Round-2 metrics use explicit names (not ambiguous 'round2_total')."""
    code = Path(REPO / "scripts/run_eval.py").read_text(encoding="utf-8")

    # Deprecated field
    assert "round2_total" not in code or "# old" in code.lower(), \
        "run_eval.py must not use ambiguous 'round2_total'"

    # Standardized fields
    required = [
        "round2_planned_queries",
        "round2_executed_queries",
        "round2_filtered_queries",
        "final_round2_papers",
    ]
    for field in required:
        assert field in code, f"run_eval.py must use standardized field '{field}'"


def test_s1_deferred_acceptance_exists():
    """Verify deferred acceptance list exists."""
    path = REPO / "eval/runs/s1/S1_DEFERRED_ACCEPTANCE.md"
    assert path.exists(), "S1_DEFERRED_ACCEPTANCE.md must exist"
    content = path.read_text(encoding="utf-8")
    assert "LLM_API_HTTP_402_INSUFFICIENT_BALANCE" in content
    assert "FAST Online Smoke Test" in content
    assert "DEEP Online Smoke Test" in content
