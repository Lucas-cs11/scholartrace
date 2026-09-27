"""v3-MVP Anchor-Augmented Query Planner（M1_ANCHOR_AUGMENTED）。

把 v3A 完整研究型规划器瘦身为单一杠杆：让 Planner 生成更有信息量、更互补的检索词，
把 raw recall 拉起来。目标修复 oracle 诊断里最大的两个失败类：
COMPOSITIONAL_QUERY_FAILURE（67）与 VOCABULARY_MISMATCH（46）。

用户批准的 v3-MVP 规范（一条不超）：
- 一次 LLM 调用：question -> {core_queries, anchor_queries, discovery_queries}。
- 最多 5 个 query，不必填满三种类型。
- CORE：高密度学术检索压缩（方法+对象，4-8 词），抓住研究问题主体。
- ANCHOR：实体锚点 <专名实体 + 任务/方法>，捕获 benchmark/dataset/model/method/arch
  等具体专名，精准定位论文（如 "HotPotQA chain of thought LLM"）。
- DISCOVERY：词汇桥 / 学术替代表达，把研究问题的口语化说法翻译成论文标题措辞
  （如 "用 RL 优化扩散模型" -> "reward optimization" / "preference alignment" /
  "policy gradient"）。
- 最小 Query Filter：exact dedup + normalized near-dedup + 去过度泛化的单概念 query。
  不做 specificity/relevance/novelty 打分（那些属于被砍的完整 QueryPortfolioV3）。

Gold isolation：plan_raw 只接收原始 question 文本，绝不接触 gold 标题/DOI/分类。

生产搜索链路零改动：intent 为 core/anchor/discovery，均非 ASSOC_INTENT（"联想论文名"），
因此不触发 search.py 的 assoc 保送 -> prekeep/reranker 完全冻结；B1_MAX_SUBQUERIES=5
天然实现「最多 5 query」上限。M1 实验脚本把 plan 以 v=2 写入 plan cache（key=query text），
_plan_and_recall 缓存命中后不再调用 parser/planner。
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from src.llm import LLMClient, LLMError
from src.schemas import SubQuery
from src.telemetry import Telemetry

# 三种 query 类型标识
INTENT_CORE = "core"
INTENT_ANCHOR = "anchor"
INTENT_DISCOVERY = "discovery"

MAX_QUERIES = 5
PRIORITY = {INTENT_CORE: 5, INTENT_ANCHOR: 4, INTENT_DISCOVERY: 3}

REVISION = 2  # ONE_PLANNER_REVISION：v1 的「词汇桥 discovery」召回 0 gold；v2 改为产出具体论文标识/标题片段，恢复 v2 assoc 的精准度。

# Rev2 设计原则：SPECIFICITY 优先于覆盖率。检索 query 的价值在于「像一个真实论文标题里的唯一串」，
# 而不是「覆盖问题的所有侧面」。宁可给孤立的强专名，也不给宽泛方法词堆砌。
SYSTEM_PROMPT = """你是学术论文检索系统的锚点增强查询规划器。任务：把一条复杂的学术研究问题，一次性生成最多 5 个可直接检索的学术检索 query。

你只能看到问题文本（一条字符串）。问题可能描述研究方向、方法偏好、对象、约束，甚至提出一个带立场的论点。

输出一个 JSON 对象：
{"core_queries": ["..."], "anchor_queries": ["..."], "discovery_queries": ["..."]}
每个数组元素是一条检索关键词串（英文，1-10 个词，不用句号）。三个数组加起来 ≤ 5 个；不必填满，宁缺毋滥。

最高原则：SPECIFICITY 优先于覆盖率。一条 query 的价值在于它像「某个真实论文标题里的唯一串」——越具体、越像一个专名/标题片段，越容易命中那篇论文本身。宁可给出孤立的强专名，也不要宽泛方法词的堆砌。

1. ANCHOR（最高优先级，产出最多）——具体论文标识：写出该主题下你确信真实存在的研究/方法/基准/模型的**具体名字**，作为独立的专名或专名+一个任务限定。目标是把检索指向具体那几篇论文。
   例："RL 优化视频扩散模型" -> "InstructVideo"（孤立专名即可，或 "InstructVideo human feedback reward"）
       "LLM 量化预训练"     -> "BitNet" / "QLoRA" / "GPTQ"
       "LLM agent 金融评测" -> "FinBen" / "FinGPT"
   专名优先：尽量给「论文标题级别的标识」，而不是 "reinforcement learning diffusion model" 这类方法词。

2. DISCOVERY（次高）——标题片段桥：输出 1-3 个「接近真实论文标题措辞」的片段，把问题的口语说法翻译成标题语言。
   例："LLM 思维链推理" -> "chain-of-thought prompting elicits reasoning" / "let's think step by step"
       "RL 对齐 LLM"     -> "aligning language models with human preferences"
   片段要像标题里会出现的短语，而不是把口语词逐个替换。

3. CORE（可选，0-1 条）——只有当 ANCHOR/DISCOVERY 都无法覆盖问题时才给：把问题主体压成一条高密度方法+对象检索串。
   例："用强化学习优化扩散模型" -> "reinforcement learning diffusion model"

规则：
- 总数 ≤ 5。优先 ANCHOR 和 DISCOVERY；CORE 是可选的 0-1 条。
- 允许单专名 query（如 "InstructVideo"、"FinBen"）——这是刻意为之，专名比宽泛词更有区分度。
- 若某问题你找不到任何真实专名/标题片段，宁可不填，也不要退化成宽泛方法词。
- 禁止纯宽泛概念（如单独 "LLM"、"language model"、"multimodal"、"neural network"、"deep learning"）。
- 只输出 JSON 对象，不要 markdown 代码块、不要任何解释。"""

# 过度泛化的单概念词：仅由这些词构成的 query 无区分度，召回对 gold 无帮助
GENERIC_TERMS = {
    "llm", "llms", "lm", "model", "models", "language", "languages", "nlp", "ml", "ai",
    "neural", "network", "networks", "deep", "learning", "machine", "multimodal",
    "agent", "agents", "method", "methods", "approach", "approaches", "framework",
    "frameworks", "paper", "papers", "research", "study", "studies", "task", "tasks",
    "technique", "techniques", "application", "applications", "survey", "overview",
    "review", "text", "image", "video", "data", "dataset", "datasets", "large",
}

# 停用词（near-dedup 与泛化检测用）
_STOPWORDS = set(
    "a an the of in on for with and or to from by at is are was were be been being this that "
    "these those as using use uses using used how what when which why where who whose can could "
    "should would may might do does did not no yes more less most their its our your my we they "
    "he she it about between among into over under against during without through plus vs versus "
    "via than then there here some any each both all also too very such".split()
)


def norm_text(t: str) -> str:
    """小写 + 压缩空白。"""
    return " ".join(t.lower().split())


def _norm_token(tok: str) -> str:
    """轻量规整（仅用于去重/近义比较，不改原始 query）：去词尾复数 s。"""
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _content_tokens(t: str) -> set[str]:
    """内容词集合（去掉停用词，轻量规整）。"""
    return {_norm_token(w) for w in re.findall(r"[a-z0-9][a-z0-9\-']*", norm_text(t)) if w not in _STOPWORDS}


def _looks_proper(tok: str) -> bool:
    """专名启发式：全大写缩略词（RLHF/GPT）或混合大小写（HotPotQA/Q-Align/BitNet）。"""
    t = tok.replace("-", "").replace("'", "")
    if not t:
        return False
    if t.isupper() and len(t) >= 2:
        return True
    return t != t.lower() and t != t.upper()


def _is_too_generic(text: str) -> bool:
    """过度泛化的单概念 query：无专名、内容词全部来自 GENERIC_TERMS，或内容词 < 2。"""
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-']*", text)
    content: list[tuple[str, str]] = []  # (原词, 小写词)，去掉停用词
    for w in tokens:
        if w.lower() not in _STOPWORDS:
            content.append((w, w.lower()))
    if not content:
        return True
    if len(content) == 1:
        # 单 token：仅专名（proper，保留原始大小写判断）值得保留；泛词（如 "attention"）丢
        return not _looks_proper(content[0][0])
    return all(lc in GENERIC_TERMS for _, lc in content)


def _near_dup(tokens: set[str], kept: list[set[str]], threshold: float = 0.7) -> bool:
    """normalized near-dedup：与已保留 query 的 token 集合 Jaccard ≥ threshold 视为重复。"""
    for kt in kept:
        union = len(tokens | kt)
        if union and len(tokens & kt) / union >= threshold:
            return True
    return False


def _fallback_subqueries(raw_query: str) -> list[SubQuery]:
    """LLM 失败/输出非法时的降级：原始问题（截断）作为唯一 core query。"""
    text = norm_text(raw_query)[:120]
    if not text:
        text = "academic paper search"
    return [SubQuery(id="sq1", query_text=text, intent=INTENT_CORE, priority=PRIORITY[INTENT_CORE])]


class AnchorAugmentedPlannerV3:
    """v3-MVP：一次 LLM 调用生成 ≤5 个互补检索 query，失败降级为原始问题。"""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient(tier="fast")

    async def plan_raw(self, raw_query: str, telemetry: Telemetry | None = None) -> list[SubQuery]:
        """question 文本 -> SubQuery 列表（intent ∈ core/anchor/discovery，≤5 条）。"""
        try:
            data = await self.llm.complete_json(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_query.strip()},
                ],
                max_tokens=1024,
                telemetry=telemetry,
                note="plan_anchor",
            )
            subs = self._build_subs(data)
            if subs:
                return subs
            raise LLMError("anchor planner 返回空 query 列表")
        except (LLMError, ValidationError) as e:
            if telemetry:
                telemetry.add_fallback(f"plan_anchor: {type(e).__name__}")
            return _fallback_subqueries(raw_query)

    def _build_subs(self, data: dict) -> list[SubQuery]:
        """{core,anchor,discovery} -> 过滤/去重 -> SubQuery 列表（≤5）。"""
        picked: list[tuple[int, str, str]] = []  # (priority, type, text)
        seen: set[str] = set()                   # normalized exact-dedup
        kept_tokens: list[set[str]] = []         # near-dedup 参考
        for qtype in (INTENT_CORE, INTENT_ANCHOR, INTENT_DISCOVERY):
            items = data.get(f"{qtype}_queries") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if len(picked) >= MAX_QUERIES:
                    break
                if not isinstance(item, str):
                    continue
                text = " ".join(item.split())
                key = norm_text(text)
                if not key or key in seen:
                    continue
                toks = _content_tokens(key)
                if not toks or _is_too_generic(key) or _near_dup(toks, kept_tokens):
                    continue
                seen.add(key)
                kept_tokens.append(toks)
                picked.append((PRIORITY[qtype], qtype, text))
            if len(picked) >= MAX_QUERIES:
                break
        if not picked:
            return []
        return [
            SubQuery(
                id=f"sq{i + 1}",
                query_text=text,
                intent=qtype,
                priority=priority,
            )
            for i, (priority, qtype, text) in enumerate(picked)
        ]
