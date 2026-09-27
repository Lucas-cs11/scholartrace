"""跨 provider 的论文规范身份（canonical identity）与 title 归一化匹配。

学术 API 用不同主键：OpenAlex 返回 W-id，Crossref/OpenCitations 返回 doi，
SemanticScholar 返回 S2-id。同一篇论文在不同 provider 下 paper_id 不同，但
DOI 或标题一致。canonical key 优先 DOI（跨 provider 统一），否则 paper_id，
否则归一化标题——供 funnel 统计、gold lifecycle、重复召回诊断使用。

注意：这是 Phase 2 观测层，不改生产去重逻辑（生产仍用 paper_id or doi）。
"""
from __future__ import annotations

import re

from src.schemas import PaperEvidence


def norm_title(text: str) -> str:
    """keep_letters 归一化（与 eval/harness.py 的 _norm_title_letters 一致）。"""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def clean_doi(doi: str) -> str:
    doi = (doi or "").strip().lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:", "doi/"):
        if doi.startswith(p):
            doi = doi[len(p):]
    return doi


def canonical_paper_id(ev: PaperEvidence) -> str:
    """论文规范主键：DOI > paper_id > 归一化标题。"""
    doi = clean_doi(ev.identity.doi)
    if doi:
        return f"doi:{doi}"
    pid = ev.identity.paper_id
    if pid:
        return pid
    t = norm_title(ev.identity.title)
    return f"title_n:{t}" if t else "unknown"


def is_gold_by_title(ev: PaperEvidence, gold_title_ns: set[str]) -> bool:
    """标题 keep_letters 归一化后是否命中 gold 集合。"""
    t = norm_title(ev.identity.title)
    return bool(t) and t in gold_title_ns


def title_n_of(ev: PaperEvidence) -> str:
    return norm_title(ev.identity.title)
