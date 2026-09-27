"""后端 API 层（FastAPI）：把 Research Engine 封装为 HTTP 服务。

前端 / iOS 通过 HTTP 调用，不直接接触检索细节。mode 选择实验链路：
b0 单查询 / b1 查询理解+子查询召回 / b2 +引文扩展 / b3 +LLM 精排 / b4 +缓存预算 / full 全链路。
"""
from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

from src.schemas import QueryIR
from src.search import SearchEngine

app = FastAPI(title="ScholarTrace Contest API", version="0.3.0")
_engine = SearchEngine()

# 前端静态资源（R2 Web Demo）
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

# R3 轻量数据库（SQLite：搜索历史）
_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scholartrace.db"


def _init_db() -> None:
    try:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS search_history ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "query TEXT NOT NULL, mode TEXT NOT NULL,"
                "result_count INTEGER DEFAULT 0, created_at TEXT NOT NULL)"
            )
    except sqlite3.Error:
        pass  # DB 故障不影响搜索服务


_init_db()

MODE_MAP = {
    "b0": _engine.search,
    "b1": _engine.search_b1,
    "b2": _engine.search_b2,
    "b3": _engine.search_b3,
    "b4": _engine.search_b4,
    "full": _engine.search_full,
}
Mode = Literal["b0", "b1", "b2", "b3", "b4", "full"]


# ------------------------------------------------------------------ 请求/响应模型
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(20, ge=1, le=50)
    mode: Mode = "full"
    summarize: bool = False   # 是否生成结果归纳（R1，结构化展示）


class PaperOut(BaseModel):
    paper_id: str
    title: str
    doi: str | None = None
    authors: list[str] = []
    venue: str | None = None
    year: int | None = None
    score: float
    label: str
    reason: str = ""
    constraint_coverage: dict[str, str] = {}


class TraceOut(BaseModel):
    round: int
    query: str = ""
    api: str = ""
    latency_ms: float = 0
    tokens: int = 0
    candidate_delta: int = 0
    relevant_delta: int = 0


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[PaperOut] = []
    telemetry: dict
    traces: list[TraceOut] = []
    summary: dict = {}   # 结果归纳（R1：query_summary/groups/overall_summary/relation_graph）


# ------------------------------------------------------------------ 异常处理
@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    import logging
    import traceback

    # 服务端记录完整堆栈（前端仍只返回类型名，不泄露内部信息）
    logging.getLogger("uvicorn.error").error(
        "Search request failed: %s %s\n%s",
        request.method,
        request.url.path,
        traceback.format_exc(),
    )
    return JSONResponse(status_code=500, content={"detail": f"internal error: {type(exc).__name__}"})


# ------------------------------------------------------------------ 端点
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "api": app.title, "version": app.version}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    t0 = time.time()
    search_fn = MODE_MAP.get(req.mode)
    results, telemetry, traces = await search_fn(req.query, top_k=req.top_k)
    latency = round((time.time() - t0) * 1000, 1)

    # R1 结果归纳：可选，模式 full 时生成结构化归纳（赛题功能点 4）
    summary: dict = {}
    if req.summarize:
        ir = QueryIR(raw_query=req.query)
        s = await _engine.summarizer.summarize(req.query, ir, results, telemetry=telemetry)
        summary = s.model_dump()

    # R3 记录搜索历史（失败不影响响应）
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO search_history (query, mode, result_count, created_at) VALUES (?,?,?,?)",
                (req.query, req.mode, len(results), datetime.utcnow().isoformat()),
            )
    except sqlite3.Error:
        pass

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results=[
            PaperOut(
                paper_id=r.paper.paper_id,
                title=r.paper.title,
                doi=r.paper.doi,
                authors=r.paper.authors,
                venue=r.paper.venue,
                year=r.paper.year,
                score=r.score,
                label=r.label.value,
                reason=r.reason,
                constraint_coverage=r.constraint_coverage,
            )
            for r in results
        ],
        telemetry={**telemetry.summary(), "endpoint_latency_ms": latency},
        traces=[TraceOut(**tr.model_dump()) for tr in traces],
        summary=summary,
    )


@app.get("/history")
async def history(limit: int = 20) -> list[dict]:
    """最近搜索历史（R3）。"""
    try:
        with sqlite3.connect(_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT id, query, mode, result_count, created_at FROM search_history "
                "ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 100)),),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {"id": r[0], "query": r[1], "mode": r[2], "result_count": r[3], "created_at": r[4]}
        for r in rows
    ]


# ------------------------------------------------------------------ 前端 Demo
@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_FRONTEND / "index.html")


if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")
