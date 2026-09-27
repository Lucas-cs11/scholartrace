"""验证假设：LLM 联想「代表性论文关键词」能否召回 gold（失败查询的召回修复方向）。

三阶段对照：
  1. oracle：gold 标题显著词直接搜 OpenAlex top-20 → 证明「搜对词就能命中」
  2. LLM 联想：DeepSeek 根据 query 联想该主题真实论文的标题关键词
  3. 联想词搜索 top-20 → 能否命中 gold

配额敏感：仅测 8 条代表性失败查询；oracle 1 词 + 联想 ≤6 词 / 条 ≈ 56 次调用（本地 IP 配额充足）。
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import LLMClient
from src.adapters.openalex import OpenAlexAdapter

FAIL_IDS = [8, 23, 25, 29, 35, 39, 42, 48]

STOP = set("""a an the of in on for to and with using based large language models paper papers study studies approach method how does which what give find show that about using can result better than from their via into over under across same different toward its own our new deep very""".split())

def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())

def sig_words(title: str, k: int = 6) -> list[str]:
    ws = [w for w in re.sub(r"[^a-z0-9]+", " ", title.lower()).split() if len(w) > 3 and w not in STOP]
    return ws[:k]

async def oa_hit(oa: OpenAlexAdapter, term: str, gold_norm: set[str]) -> tuple[bool, str]:
    try:
        evs = await oa.search(term, limit=20)
    except Exception as e:
        return False, f"ERR:{type(e).__name__}"
    for ev in evs:
        tn = norm(ev.identity.title)
        if tn in gold_norm:
            return True, ev.identity.title
    return False, ""

async def main() -> None:
    data = {}
    for line in open("data/benchmarks/pasa/RealScholarQuery/test.jsonl", encoding="utf-8"):
        d = json.loads(line)
        data[d["qid"]] = d

    queries = []
    for i in FAIL_IDS:
        d = data[f"RealScholarQuery_{i}"]
        queries.append({
            "qid": i,
            "question": d["question"],
            "golds": [a for a in (d.get("answer") or []) if a],
            "gold_norm": {norm(t) for t in (d.get("answer") or []) if t},
        })

    llm = LLMClient(tier="fast")
    oa = OpenAlexAdapter(mailto="lujie.cs@gmail.com")
    try:
        # ---- 阶段 2：LLM 联想（一次性给全部 8 条）----
        prompt_qs = "\n\n".join(f"[{i}] {q['question']}" for i, q in enumerate(queries))
        sys_p = """你是熟悉 AI 领域论文的专家。下面给出 8 个研究主题查询（[0]-[7]）。
对每个主题，列出你认为该主题下【真实存在、有影响力的具体论文】的标题关键词——优先专名（方法名/模型名/数据集名/基准名，如 "Curry-DPO"、"MUSTARD"、"Q-Align"、"BitNet"），也可是经典论文标题的简短片段（2-5 个词，如 "video diffusion reward gradients"）。
不要泛化概念描述（如 "multimodal LLM" 太泛，要具体论文名）。
每个主题给 6 个关键词串，每行一个，格式必须是严格 JSON 数组，如 [["term1","term2",...], ...]，8 个内层数组对应对应主题，不要任何其他文字。"""
        text = await llm.complete(
            [{"role": "system", "content": sys_p},
             {"role": "user", "content": prompt_qs}],
            max_tokens=2048, note="assoc_recall_probe",
        )
        # 兼容 {"terms": [...]} / 直接数组
        text = text.strip()
        if not text.startswith("["):
            text = text[text.find("["):]
        if text.endswith("```"):
            text = text[:-3]
        raw = json.loads(text)
        assoc = raw if isinstance(raw, list) else (raw.get("terms") or raw.get("associations") or [])
        print(f"LLM 联想解析: {len(assoc)} 条")

        print(f"\n{'qid':>4} | {'oracle命中':<6} {'联想命中':<6} | gold 标题")
        print("-" * 100)
        oracle_hits = assoc_hits = 0
        for idx, q in enumerate(queries):
            gold0 = q["golds"][0]
            # ---- 阶段 1：oracle - gold 标题显著词 ----
            oterm = " ".join(sig_words(gold0, 4))
            o_hit, o_title = await oa_hit(oa, oterm, q["gold_norm"])
            # ---- 阶段 3：LLM 联想词逐个搜 ----
            terms = assoc[idx] if idx < len(assoc) else []
            a_hit = False; a_title = ""
            for t in terms:
                if not isinstance(t, str) or not t.strip():
                    continue
                hit, hit_title = await oa_hit(oa, t.strip(), q["gold_norm"])
                if hit:
                    a_hit, a_title = True, hit_title
                    break
            if o_hit: oracle_hits += 1
            if a_hit: assoc_hits += 1
            print(f"{q['qid']:>4} | {str(o_hit):<6} {str(a_hit):<6} | {gold0[:60]}")
            print(f"      oracle词: {oterm!r}")
            print(f"      联想词: {terms}")
            if a_hit:
                print(f"      → 联想命中: {a_title[:70]}")
            elif not o_hit:
                print(f"      → oracle 未命中（gold 可能不在 OpenAlex 或词序敏感）")

        print("-" * 100)
        print(f"oracle 命中 {oracle_hits}/{len(queries)}（搜对词即可命中）")
        print(f"LLM 联想命中 {assoc_hits}/{len(queries)}（当前 pipeline 为 0）")
    finally:
        await llm.close()
        await oa.close()


if __name__ == "__main__":
    asyncio.run(main())
