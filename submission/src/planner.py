"""子查询分解与查询改写：QueryIR -> 多个可独立检索的 SubQuery。

动机（B0 基线结论）：冗长自然语言查询直接检索 OpenAlex 召回率极低（F1=0.0095）。
把查询分解为多个检索友好的关键词串是提升召回的第一杠杆。

设计：
- 一次 LLM 调用完成分解+改写（省 token）。
- 每个 SubQuery 是可直接执行的检索关键词串，intent 说明为什么搜。
- LLM 失败/输出非法时用 IR 字段拼凑降级子查询——B1 链路永不空手。
"""
from __future__ import annotations

from pydantic import ValidationError

from src.llm import LLMClient, LLMError
from src.schemas import QueryIR, SubQuery
from src.telemetry import Telemetry

SYSTEM_PROMPT = """你是学术论文检索系统的查询规划器。把解析好的查询结构（JSON）分解成 2-5 个可独立检索的子查询，每个子查询是学术搜索引擎（OpenAlex）的一次检索关键词串。

输入是 JSON 对象：{"raw_query": ..., "topic": ..., "entities": [...], "methods": [...], "datasets": [...], "must_constraints": [...], "should_constraints": [...], ...}

输出 JSON 对象：{"subqueries": [{"query_text": str, "intent": str, "priority": int, "parent_constraint_ids": [int]}]}
parent_constraint_ids 可选：该子查询主要覆盖的 must_constraints / should_constraints 的下标（must 下标从 0 开始，should 紧随其后偏移 must 的数量）。未命中任何约束则省略。

分解策略：
1. 专名优先：每个命名实体（模型/方法/数据集，如 "PointNet"、"BERT"、"ShapeNet"）各成一个子查询，可叠加 1-2 个最相关的限定词（如 "PointNet 3D point cloud classification"）。
2. 标题关键词兜底：对研究主题里最核心的 1-2 个概念，提炼出【最可能出现在经典论文标题中】的简短关键词（2-4 个词），单独成子查询。经典论文标题通常简短专名化（如 "Attention Is All You Need"、"Denoising Diffusion Probabilistic Models"），与任务描述的长句词汇不重叠——单独关键词检索比长 AND 组合更易命中标题。例子：主题"transformer 机器翻译"→ "attention" 或 "transformer"；"扩散模型"→ "denoising diffusion"。
3. 主题兜底：一个覆盖研究主题的宽子查询（4-8 个关键词），保证没有专名时也有召回。
4. 交叉组合：需要时做 实体 × 关键约束 的组合（如 "BERT fine-tuning"）。
5. query_text 是检索关键词串：保留专名和领域术语，去掉虚词和冗长描述；英文 1-10 个词。标题关键词子查询可以很短（1-3 个词）。
6. intent 说明为什么搜这个子查询（如 代表术语 / 标题关键词 / 主题覆盖 / 约束限定）。
7. priority 1-5：专名和标题关键词 4-5，主题宽查询 2-3，交叉组合 3-4，兜底 1。
8. 只输出 JSON 对象，不要 markdown 代码块、不要任何解释。

额外输出字段 assoc_terms（JSON 数组，恰好 6 个字符串）：联想该研究主题下【真实存在、能定位到某一篇具体论文】的检索词。每个词串要么是论文标题里的专名缩写（方法名/模型名/数据集名/基准名，如 "Q-Align"、"MUSTARD"、"Curry-DPO"、"BitNet"、"InstructVideo"），要么是标题的独特片段组合（方法+任务，如 "video diffusion reward gradients"、"1-bit LLM"、"sparse reward agent"）。要求：
- 窄到能区分单篇论文（"Q-Align video aesthetics" 好；"multimodal LLM" 差；仅代表大方向的知名模型名如 "Sora" 除非强相关否则避免）。
- 不要泛化概念描述；专名越具体越好。联想优先于主题泛化。
示例（要这种具体度）：主题"用强化学习优化视频生成" → ["InstructVideo human feedback", "video diffusion reward gradients", "Align-A-Video"]；主题"低比特量化预训练" → ["BitNet 1.58 bits", "1-bit LLM", "LoQT rank adapters"]。"""


ASSOC_INTENT = "联想论文名"
MAX_ASSOC_TERMS = 6
PLANNER_VERSION = 2  # plan cache 版本；planner 策略变更时 +1 使旧缓存失效


def fallback_subqueries(ir: QueryIR) -> list[SubQuery]:
    """无 LLM 可用时的降级分解：直接由 IR 字段拼检索串。"""
    subs: list[SubQuery] = []
    n = 1
    for e in ir.entities:
        subs.append(SubQuery(id=f"sq{n}", query_text=e, intent="代表术语", priority=4))
        n += 1
    if ir.methods:
        subs.append(SubQuery(id=f"sq{n}", query_text=" ".join(ir.methods[:4]), intent="方法约束", priority=2))
        n += 1
    if not subs and ir.raw_query.strip():
        subs.append(SubQuery(id="sq1", query_text=ir.raw_query.strip()[:120], intent="原始查询", priority=1))
    return subs


class SubQueryPlanner:
    """LLM 驱动的子查询分解与改写，失败降级为 IR 拼凑。"""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(tier="fast")

    async def plan(self, ir: QueryIR, telemetry: Telemetry | None = None) -> list[SubQuery]:
        try:
            data = await self.llm.complete_json(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ir.model_dump_json(exclude_none=True)},
                ],
                max_tokens=1536,
                telemetry=telemetry,
                note="plan_subqueries",
            )
            subs = self._validate(data.get("subqueries") or [])
            # 联想论文名：LLM 先验回忆该主题真实存在的代表性论文关键词（召回修复关键杠杆）
            subs += self._parse_assoc(data.get("assoc_terms") or [])
            if subs:
                return subs
            raise LLMError("planner 返回空子查询列表")
        except (LLMError, ValidationError) as e:
            if telemetry:
                telemetry.add_fallback(f"plan_subqueries: {type(e).__name__}")
            return fallback_subqueries(ir)

    def _parse_assoc(self, raw: list) -> list[SubQuery]:
        """assoc_terms 转高优先级子查询（intent=ASSOC_INTENT），去重、截断。"""
        out: list[SubQuery] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(
                SubQuery(
                    id=f"assoc{len(out) + 1}",
                    query_text=text,
                    intent=ASSOC_INTENT,
                    priority=5,
                )
            )
            if len(out) >= MAX_ASSOC_TERMS:
                break
        return out

    def _validate(self, raw: list) -> list[SubQuery]:
        """清洗 LLM 输出：过滤空 query_text、去重、id 重排、priority 截断、约束链解析。"""
        subs: list[SubQuery] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("query_text", "")).strip()
            if not text:
                continue
            key = " ".join(text.lower().split())
            if key in seen:
                continue
            seen.add(key)
            try:
                priority = int(item.get("priority", 1))
            except (TypeError, ValueError):
                priority = 1
            # 约束链：parent_constraint_ids（非负整数，去重排序）
            parent_ids: list[int] = []
            for cid in item.get("parent_constraint_ids") or []:
                if isinstance(cid, bool):
                    continue
                try:
                    cid = int(cid)
                except (TypeError, ValueError):
                    continue
                if cid >= 0 and cid not in parent_ids:
                    parent_ids.append(cid)
            subs.append(
                SubQuery(
                    id=f"sq{len(subs) + 1}",
                    query_text=text,
                    intent=str(item.get("intent", "")).strip(),
                    priority=max(1, min(5, priority)),
                    parent_constraint_ids=parent_ids,
                )
            )
        return subs
