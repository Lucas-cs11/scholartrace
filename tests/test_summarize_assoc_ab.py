import json

import pytest

from scripts.summarize_assoc_ab import summarize


def _write_report(path, *, f1, precision, recall, api_calls):
    path.write_text(json.dumps({
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "api_calls": api_calls,
    }), encoding="utf-8")


def test_summarize_aggregates_reports(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first, f1=0.2, precision=0.1, recall=0.5, api_calls=4)
    _write_report(second, f1=0.0, precision=0.0, recall=0.0, api_calls=2)

    assert summarize([str(first), str(second)]) == {
        "queries": 2,
        "mean_f1": 0.1,
        "mean_precision": 0.05,
        "mean_recall": 0.25,
        "f1_positive": 1,
        "f1_ge_01": 1,
        "f1_ge_02": 1,
        "total_api_calls": 6,
        "mean_api_calls": 3.0,
    }


def test_summarize_rejects_empty_report_list():
    with pytest.raises(ValueError, match="no reports"):
        summarize([])
