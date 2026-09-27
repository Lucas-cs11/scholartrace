"""统一评测 harness。

职责：加载挑战 query -> 运行搜索 -> 计算 P/R/F1 -> 落盘 RunReport -> 输出汇总。
所有模块改动必须通过这里验证，禁止"感觉更智能"式改动。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Awaitable, Callable

from src.schemas import RankLabel, RankResult, RunReport, SearchTrace
from src.telemetry import Telemetry

SearchFn = Callable[[str], Awaitable[tuple[list[RankResult], Telemetry, list[SearchTrace]]]]


def _validate_results(results: list[RankResult]) -> bool:
    """结构化输出校验：paper_id 非空、score 在 [0,1]、label 合法。"""
    for r in results:
        if not r.paper.paper_id:
            return False
        if not (0.0 <= r.score <= 1.0):
            return False
        if not isinstance(r.label, RankLabel):
            return False
    return True


def _norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    return doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def _norm_title(title: str) -> str:
    return " ".join(title.lower().split())


def _norm_title_letters(title: str) -> str:
    """keep_letters 归一化（PaSa 基准用）：仅保留字母数字，小写。

    对基准标题更鲁棒：容错标点/空白/unicode 差异（如 em-dash vs hyphen）。
    """
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def load_challenges(path: str | Path) -> list[dict]:
    challenges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                challenges.append(json.loads(line))
    return challenges


def match_gold(challenge: dict) -> list[set[str]]:
    """返回每篇 gold 论文的标识组。组内任一标识命中即视为命中该论文。

    以「论文」为计数单位，避免 openalex_id/doi/title 三个 key 虚增 gold 总数。
    """
    groups: list[set[str]] = []
    for g in challenge.get("gold", []):
        keys = set()
        oid = g.get("openalex_id")
        if oid:
            keys.add(f"openalex:{oid}")
        doi = _norm_doi(g.get("doi") or g.get("openalex_doi"))
        if doi:
            keys.add(f"doi:{doi}")
        if g.get("title"):
            keys.add(f"title:{_norm_title(g['title'])}")
            keys.add(f"title_n:{_norm_title_letters(g['title'])}")
        if keys:
            groups.append(keys)
    return groups


def _paper_keys(r: RankResult) -> set[str]:
    keys = set()
    if r.paper.paper_id:
        keys.add(f"openalex:{r.paper.paper_id}")
    doi = _norm_doi(r.paper.doi)
    if doi:
        keys.add(f"doi:{doi}")
    if r.paper.title:
        keys.add(f"title:{_norm_title(r.paper.title)}")
        keys.add(f"title_n:{_norm_title_letters(r.paper.title)}")
    return keys


def compute_p_r_f1(predicted: list[RankResult], gold_groups: list[set[str]]) -> dict:
    tp = 0
    matched_groups: set[int] = set()
    for r in predicted:
        pkeys = _paper_keys(r)
        for gi, gkeys in enumerate(gold_groups):
            if gi in matched_groups:
                continue
            if pkeys & gkeys:
                tp += 1
                matched_groups.add(gi)
                break

    gold_n = len(gold_groups)
    pred_n = len(predicted)
    precision = tp / pred_n if pred_n else 0.0
    recall = tp / gold_n if gold_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": max(pred_n - tp, 0),
        "fn": max(gold_n - tp, 0),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


async def evaluate(
    search_fn: SearchFn,
    gold_path: str,
    runs_dir: str,
    experiment_id: str = "B0",
    top_k: int = 20,
    limit: int | None = None,
    skip_on_error: bool = False,
    after_query: Callable[[], None] | None = None,
    raise_on_error: Callable[[Exception], bool] | None = None,
) -> list[RunReport]:
    challenges = load_challenges(gold_path)
    if limit:
        challenges = challenges[:limit]

    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    reports: list[RunReport] = []

    for i, ch in enumerate(challenges, 1):
        query = ch["query"]
        t0 = time.time()
        error_note = ""
        try:
            results, telemetry, traces = await search_fn(query, top_k=top_k)
        except Exception as e:
            if raise_on_error and raise_on_error(e):
                raise
            if not skip_on_error:
                raise
            error_note = f"{type(e).__name__}: {e}"
            results, telemetry, traces = [], Telemetry(), [
                SearchTrace(round=0, query=query[:80], api="error", candidate_delta=0)
            ]
        latency_ms = round((time.time() - t0) * 1000, 1)

        gold_groups = match_gold(ch)
        metrics = compute_p_r_f1(results, gold_groups)

        report = RunReport(
            experiment_id=experiment_id,
            query_id=ch.get("query_id", f"q{i}"),
            raw_query=query,
            gold_ids=sorted(k for g in gold_groups for k in g),
            predicted_ids=[r.paper.paper_id for r in results],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            api_calls=telemetry.api_calls,
            llm_calls=telemetry.llm_calls,
            input_tokens=telemetry.input_tokens,
            output_tokens=telemetry.output_tokens,
            latency_ms=latency_ms,
            schema_valid=_validate_results(results),
            traces=traces,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        # 逐条落盘
        out = Path(runs_dir) / f"{experiment_id}_{report.query_id}.json"
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        reports.append(report)
        flag = f" ERROR({error_note})" if error_note else ""
        print(f"[{i}/{len(challenges)}] {report.query_id}: F1={report.f1:.4f} "
              f"P={report.precision:.4f} R={report.recall:.4f} api={report.api_calls} {latency_ms}ms{flag}")
        if after_query:
            after_query()  # 供外部逐条落盘缓存（配额跨天续跑）

    return reports


def summarize(reports: list[RunReport]) -> dict:
    n = len(reports)
    if not n:
        return {}
    agg = {
        "experiment": reports[0].experiment_id,
        "queries": n,
        "mean_f1": round(sum(r.f1 for r in reports) / n, 4),
        "mean_precision": round(sum(r.precision for r in reports) / n, 4),
        "mean_recall": round(sum(r.recall for r in reports) / n, 4),
        "total_api_calls": sum(r.api_calls for r in reports),
        "total_tokens": sum(r.total_tokens for r in reports),
        "mean_latency_ms": round(sum(r.latency_ms for r in reports) / n, 1),
    }
    agg["queries_f1_ge_05"] = sum(1 for r in reports if r.f1 >= 0.5)
    return agg
