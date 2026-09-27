"""外部基准评测：加载 PaSa RealScholarQuery 等，跑引擎评测，输出 F1 / recall@k。

数据格式（PaSa RealScholarQuery，jsonl，2025 版）：
    {"question": "...", "answer": ["期望论文标题", ...],
     "answer_arxiv_id": ["2309.04564", ...], "qid": "..."}
gold 按论文标题 keep_letters 归一化匹配（harness 新增 title_n 键）。
--skip-on-error 逐条容错 + 逐条落盘缓存，配额耗尽可跨天续跑。

用法：
    python scripts/eval_benchmark.py --data data/benchmarks/pasa/RealScholarQuery/test.jsonl \
        --mode full --top-k 20 --experiment PASA_FULL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import evaluate, summarize
from src.search import SearchEngine


def norm_title(t: str) -> str:
    """PaSa keep_letters 归一化：仅保留字母数字，小写。"""
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def load_pasa(path: str | Path) -> list[dict]:
    """把 PaSa RealScholarQuery/AutoScholarQuery 转成 harness gold 格式。

    真实 jsonl 格式（2025 版 PaSa 数据集）：
        {"question": "...", "answer": ["标题", ...],
         "answer_arxiv_id": ["2309.04564", ...], "qid": "RealScholarQuery_0"}
    gold 保留 title + arxiv_id（标题用 keep_letters 归一化匹配，arxiv_id 留作身份参考）。
    """
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            answers = d.get("answer") or []
            arxiv_ids = d.get("answer_arxiv_id") or []
            gold: list[dict] = []
            for j, a in enumerate(answers):
                if not a:
                    continue
                item: dict = {"title": a}
                if j < len(arxiv_ids) and arxiv_ids[j]:
                    item["arxiv_id"] = arxiv_ids[j]
                gold.append(item)
            out.append({
                "query_id": str(d.get("qid") or f"q{i:03d}"),
                "query": d.get("question", ""),
                "gold": gold,
            })
    return out


def to_harness_gold(queries: list[dict]) -> str:
    """写成临时 harness gold jsonl。"""
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for q in queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
    f.close()
    return f.name


def filter_existing(queries: list[dict], runs_dir: str, experiment: str) -> list[dict]:
    """断点续跑：跳过已落盘报告的 query（reports 幂等，避免重跑耗配额）。"""
    out = []
    skipped = 0
    for q in queries:
        report = Path(runs_dir) / f"{experiment}_{q['query_id']}.json"
        if report.exists():
            skipped += 1
            continue
        out.append(q)
    if skipped:
        print(f"跳过已完成的 {skipped} 条（{runs_dir}），续跑剩余 {len(out)} 条")
    return out


def _is_openalex_quota_error(exc: Exception) -> bool:
    """判断是否 OpenAlex 配额/服务暂不可用（429/503 且来自 api.openalex.org）。"""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    host = exc.request.url.host if exc.request and exc.request.url else ""
    return host == "api.openalex.org" and exc.response.status_code in (429, 503)


class QuotaExhausted(RuntimeError):
    """OpenAlex 配额/服务不可用（429/503）。外层捕获后保存进度、exit 42 提示切 IP。"""


async def _with_quota_exit(fn):
    """OpenAlex 配额退出包装器：429/503 且来自 api.openalex.org 时抛 QuotaExhausted。

    配合 --skip-existing 逐条缓存：配额耗尽即停（exit 42），换 IP/等恢复后
    重跑同一命令自动续跑。其他源（Crossref/S2/OpenCitations）的 429/503
    不退出，交给 harness skip_on_error 逐条容错。
    """

    async def wrapped(query: str, top_k: int) -> tuple:
        try:
            return await fn(query, top_k=top_k)
        except httpx.HTTPStatusError as e:
            if _is_openalex_quota_error(e):
                print(f"[{query[:50]}] OpenAlex 配额/服务不可用 ({e.response.status_code})，"
                      f"保存进度并退出。请切换 VPN 节点后重跑（--skip-existing 续跑）。", flush=True)
                raise QuotaExhausted from e
            raise

    return wrapped


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="benchmark jsonl 路径")
    parser.add_argument("--mode", default="full", choices=["b0", "b1", "b2", "b3", "b4", "full"])
    parser.add_argument("--recall-source", default=None,
                        choices=["openalex", "crossref", "s2"],
                        help="主召回源（默认取 .env RECALL_SOURCE；crossref/s2 免 OpenAlex 配额）")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enable-s2-recall", action="store_true",
                        help="方向1：子查询并行走 OpenAlex + S2 双路召回")
    parser.add_argument("--no-citation", action="store_true",
                        help="关闭多轮引文扩展（隔离方向1 召回，避免 S2 key 激活引文路径混淆）")
    parser.add_argument("--reranker-tldr", action="store_true",
                        help="方向4：精排 prompt 注入 S2 tldr 语义摘要（需 S2 召回候选带 tldr）")
    parser.add_argument("--citation-rounds", type=int, default=2,
                        help="方向5：引文扩展轮数/深度（默认2；3 探索引文-of-引文第2跳）")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过 runs_dir 已落盘报告的 query（断点续跑，省配额）")
    parser.add_argument("--runs", default="eval/runs")
    parser.add_argument("--cache", default="eval/runs/_recall_cache.jsonl")
    parser.add_argument("--plan-cache", default="eval/runs/_plan_cache.jsonl")
    args = parser.parse_args()

    experiment = args.experiment or f"{args.mode.upper()}_BENCH"
    queries = load_pasa(args.data)
    print(f"加载 {len(queries)} 条 benchmark 查询（来自 {args.data}）")
    if args.skip_existing:
        queries = filter_existing(queries, args.runs, experiment)
    if not queries:
        print("所有 query 均已完成，无需续跑")
        return
    gold_path = to_harness_gold(queries)

    # 可插拔召回源：--recall-source 覆盖 .env RECALL_SOURCE（其余配置仍从 .env 读）
    from config.settings import Settings
    from src.ranker import LLMReranker
    overrides = {}
    if args.recall_source:
        overrides["recall_source"] = args.recall_source
    if args.enable_s2_recall:
        overrides["enable_s2_recall"] = True
    need_cfg = args.recall_source or args.enable_s2_recall or args.no_citation
    reranker_kwargs = {"use_tldr": True} if args.reranker_tldr else {}
    if need_cfg or args.reranker_tldr or args.citation_rounds != 2:
        engine = SearchEngine(cfg=Settings(**overrides) if need_cfg else None,
                              enable_citation_expansion=not args.no_citation,
                              reranker=LLMReranker(**reranker_kwargs),
                              citation_max_rounds=args.citation_rounds)
        if args.recall_source:
            print(f"主召回源: {args.recall_source}")
        if args.enable_s2_recall:
            print(f"多源召回: 已启用 S2 并行召回")
        if args.no_citation:
            print(f"引文扩展: 已关闭（--no-citation）")
        if args.reranker_tldr:
            print(f"精排语义: 已启用 S2 tldr（--reranker-tldr）")
    else:
        engine = SearchEngine()
    engine.load_plan_cache(args.plan_cache)
    engine.load_recall_cache(args.cache)
    search_fn = {
        "b0": engine.search, "b1": engine.search_b1, "b2": engine.search_b2,
        "b3": engine.search_b3, "b4": engine.search_b4, "full": engine.search_full,
    }[args.mode]
    # OpenAlex 配额退出（429/503 且来自 openalex 时保存进度退出；其他源不退出）
    search_fn = await _with_quota_exit(search_fn)

    try:
        reports = await evaluate(
            lambda q, top_k: search_fn(q, top_k=top_k),
            gold_path, args.runs, experiment_id=experiment,
            top_k=args.top_k, limit=args.limit,
            skip_on_error=True,  # 非配额错误逐条容错，不中断全量
            raise_on_error=lambda e: isinstance(e, QuotaExhausted),
            after_query=lambda: (engine.save_recall_cache(args.cache),
                                 engine.save_plan_cache(args.plan_cache)),
        )
    except QuotaExhausted:
        engine.save_recall_cache(args.cache)
        engine.save_plan_cache(args.plan_cache)
        print("\n[QUOTA_EXHAUSTED] 进度已保存。请切换 VPN 节点后重跑同一命令"
              "（--skip-existing 自动续跑）。", flush=True)
        sys.exit(42)
    engine.save_recall_cache(args.cache)
    engine.save_plan_cache(args.plan_cache)
    summary = summarize(reports)
    print("\n=== 基准评测汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    summary_out = Path(__file__).resolve().parent.parent / "eval" / "runs" / "bench_summary.json"
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {summary_out}")


if __name__ == "__main__":
    asyncio.run(main())
