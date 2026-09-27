"""S1 replay demo：在 Q29 / Q43 上离线回放 FAST 与 DEEP，展示完整 SearchTrace。

    python3 scripts/replay_demo.py                 # 全部 4 种组合（fast/deep × Q29/Q43）
    python3 scripts/replay_demo.py --qid 29 --mode deep

Q29/Q43 是冻结 22 查询集中 evidence-guided Round-2 有增量 Gold 的样例：
- Q29（entity）：follow-up "Lean" → DeepSeek-Prover，rank 8
- Q43（gap）：   follow-up → antibody DPO 论文，rank 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from s1.config import load_config  # noqa: E402
from s1.pipeline import ContestEngine  # noqa: E402

DEMO = {
    "RealScholarQuery_29": "Research on teaching llms to do math prove and solve IMO level math problems.",
    "RealScholarQuery_43": "AI for Science papers, especially protein design and DPO of antibody design.",
}
EVIDENCE_KEYS = ["title", "year", "venue", "source_query", "pre_rerank_rank"]


def _show_trace(res, show_r2: bool) -> None:
    t = res.trace
    print(f"\n  mode={res.mode.upper()}  qid={res.query_id}")
    print(f"  planner_version={t.planner_version}  prompt_hash={t.prompt_hash}")
    print(f"  generated_queries ({len(t.generated_queries)}):")
    for g in t.generated_queries[:6]:
        print(f"    R{g['round']} [{g['intent']:12}] {g['query']}")
    obs = t.round1_observation
    print(f"  round1: retrieved={obs.total_retrieved} dedup={obs.deduplicated_candidates} "
          f"evidence_top={len(obs.evidence_papers)}")
    if show_r2:
        dec = t.round2_decision
        print(f"  round2 continue={dec.continue_search} ({dec.continue_reason[:60]}...)")
        for fu in dec.followups:
            print(f"    fup [{fu.source:12}] status={fu.status:9} {fu.query}")
        if t.newly_discovered_papers:
            print(f"  round2 new papers: {t.newly_discovered_papers[:5]}")
    print(f"  final ranked ({len(res.results)}):  [label  score R{round} source_query]")
    for r in res.results[:8]:
        print(f"    [{r.relevance_label:6} {r.relevance_score:+.3f} R{r.retrieval_round}] "
              f"{r.source_query[:44]:44} | {r.title[:60]}")
    print(f"  cost: api={t.api_calls} llm={t.llm_calls} tokens={t.input_tokens + t.output_tokens} "
          f"latency_ms={t.total_latency_ms}")


async def _main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.mode:
        cfg.mode = args.mode.lower()
    engine = ContestEngine(cfg)

    qids = ([f"RealScholarQuery_{args.qid}"] if args.qid else list(DEMO))
    for qid in qids:
        question = DEMO.get(qid, qid)
        print(f"\n########## {qid} ##########")
        print(f"Q: {question}")
        # FAST
        cfg.mode = "fast"
        res_f = await engine.search(question, query_id=qid, mode="fast")
        _show_trace(res_f, show_r2=False)
        # DEEP（同一查询重跑 Round1 → Round2 → Merge → Rerank）
        cfg.mode = "deep"
        res_d = await engine.search(question, query_id=qid, mode="deep")
        _show_trace(res_d, show_r2=True)

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            payload = {"fast": res_f.trace.model_dump(), "deep": res_d.trace.model_dump()}
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[S1:replay_demo] 已写入 {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="S1 Q29/Q43 replay demo")
    p.add_argument("--qid", type=int, default=None)
    p.add_argument("--mode", choices=["fast", "deep"], default=None)
    p.add_argument("--config", default=str(REPO / "configs" / "deep.yaml"))
    p.add_argument("--out", default=None)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
