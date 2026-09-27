"""诊断 v2：按标题判断 gold 是否进入候选集（容忍 OpenAlex 重复 ID）。"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search import SearchEngine


def norm(t: str) -> str:
    t = (t or "").lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


CASES = [
    ("q003", "transformer architecture using self-attention for neural machine translation, sequence-to-sequence without recurrence", "Attention Is All You Need"),
    ("q005", "denoising diffusion probabilistic models for high-quality image generation, score-based generative modeling", "Denoising Diffusion Probabilistic Models"),
    ("q001", "3D point cloud classification and segmentation with deep learning networks trained on ShapeNet", "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"),
    ("q002", "deep reinforcement learning agent that achieves human-level control playing Atari games from raw pixels", "Human-level control through deep reinforcement learning"),
    ("q004", "pre-trained bidirectional transformer encoder for natural language understanding fine-tuning BERT", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
]


async def main() -> None:
    engine = SearchEngine()
    for qid, q, gold_title in CASES:
        gold_norm = norm(gold_title)
        cands = await engine.openalex.search(q, limit=50)
        found = [c for c in cands if norm(c.identity.title) == gold_norm]
        ranked = engine._lexical_rank(q, cands)
        top20 = ranked[:20]
        in_top20 = any(norm(r.paper.title) == gold_norm for r in top20)
        print(f"[{qid}] 候选={len(cands)} 按标题在候选={len(found)>0} 在Top20={in_top20}")
        if found:
            f = found[0]
            print(f"      gold实际ID={f.identity.paper_id} doi={f.identity.doi}")


if __name__ == "__main__":
    asyncio.run(main())
