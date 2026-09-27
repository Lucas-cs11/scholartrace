"""gold 挑战集校验与归一化。

把每条 gold 论文对齐到 OpenAlex 的真实身份（openalex_id / doi / title），
使评测能诚实匹配。用法：
    python scripts/curate_gold.py --src eval/gold/challenges_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import httpx


def norm_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def similarity(a: str, b: str) -> float:
    a_tok, b_tok = set(a.split()), set(b.split())
    if not a_tok or not b_tok:
        return 0.0
    return len(a_tok & b_tok) / min(len(a_tok), len(b_tok))


async def find_openalex(client: httpx.AsyncClient, title: str) -> dict | None:
    q = " ".join(title.split())[:80]
    url = "https://api.openalex.org/works"
    params = {"search": q, "per-page": 5, "select": "id,title,doi,publication_year"}
    resp = await client.get(url, params=params)
    if resp.status_code != 200:
        return None
    results = resp.json().get("results", [])
    best = None
    best_sim = 0.0
    for w in results:
        sim = similarity(title, w.get("title") or "")
        if sim > best_sim:
            best_sim = sim
            best = w
    if best is None or best_sim < 0.7:
        return None
    return best


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="eval/gold/challenges_v1.jsonl")
    parser.add_argument("--mailto", default="")
    args = parser.parse_args()

    path = Path(args.src)
    challenges = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    params = {"mailto": args.mailto} if args.mailto else {}

    async with httpx.AsyncClient(timeout=20, params=params) as client:
        for c in challenges:
            for g in c.get("gold", []):
                if g.get("openalex_id"):
                    continue
                found = await find_openalex(client, g.get("title", ""))
                if found:
                    g["openalex_id"] = found["id"].rsplit("/", 1)[-1]
                    g["verified_title"] = found.get("title")
                    g["openalex_doi"] = (found.get("doi") or "").replace("https://doi.org/", "")
                    print(f"  OK  {c['query_id']}: {g.get('title','')[:50]} -> {g['openalex_id']}")
                else:
                    print(f"  MISS {c['query_id']}: {g.get('title','')[:50]} (需人工核对)")
                time.sleep(0.1)

    out = "\n".join(json.dumps(c, ensure_ascii=False) for c in challenges) + "\n"
    path.write_text(out, encoding="utf-8")
    print(f"\n已更新 {path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
