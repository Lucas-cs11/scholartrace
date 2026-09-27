"""M3_SPARSE_PLAN_RESCUE：稀疏查询救援规划器（SparsePlanRescue）。

目标（M3 唯一杠杆）：只修 15 条 sparse plan（assoc_count==0 或 total_subquery_count<=1，即
只有 1 条 thin 子查询、几乎 0 raw gold 的 query）的 Query Formulation；7 条 rich plan 冻结
原样（含 v2 assoc 联想词锚点，是 raw=22 的来源）。**不重做整个 Planner。**

与 M1（M1_ANCHOR_AUGMENTED，全量 22 条重做、≤5 total 上限压死密度 → raw 崩到 7）的区别：
M3 只给 sparse query 补充缺失的查询密度（≤4），不动 rich query 的 assoc 锚点密度。

用户批准的 M3 规范（一条不超）：
- 一次 LLM 调用 / sparse query，≤4 条 query：CORE×1 + ANCHOR×1-2 + DISCOVERY×0-1。不必填满。
  - CORE：把复杂问题压缩成学术数据库核心检索表达式（方法+对象，2-6 词）。
  - ANCHOR：优先问题中明确/可推断的 method/model/benchmark/dataset/task 实体 →
            "<实体> + <任务/方法>" 检索表达式。
  - DISCOVERY：词汇不匹配/组合失效的替代学术措辞（同义改写/领域术语桥），0-1 条。
- 最小 Query Filter：exact dedup + normalized near-dedup + 去过度泛化单概念 query。
  不做 specificity/relevance/novelty 打分（那些属于被砍的完整 QueryPortfolioV3）。
- 冻结清单：assoc_safepass=True（KEEP_SAFEPASS）、Citation/Reference/Metadata OFF、
  OpenAlex 不变、top_k=20、Prekeep 不变、Reranker 不变、7 条 rich plan 原样。
- 无 adaptive retrieval / early stop / multi-provider / reranker 优化。

Gold isolation（EXPERIMENT_INVALID 门禁）：
- plan_raw 只接收原始 question 文本；绝不接触 gold 标题/作者/DOI/arXiv ID/摘要。
- **严禁**：基于 Gold 生成具体论文标题/作者/DOI/arXiv ID。只能使用问题文本里明确给出或
  可合理推断的实体；不确定真实存在的实体宁可不写。
- gold 只在评测阶段由 TraceRecorder 读取。

引擎加载：rescue plan 以 v=2 写入 plan cache（key=query text，ir 复用 v2 冻结 ir 以保持
reranker 看到的 query+ir 与 B0 完全一致）；_plan_and_recall 缓存命中后不再调用 parser/planner。
intent ∈ core/anchor/discovery，均非 ASSOC_INTENT，不触发 assoc 保送（对 sparse query 正常
走 prekeep；rich query 的 assoc 锚点仍按 assoc_safepass=True 保送）。
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from src.llm import LLMClient, LLMError
from src.schemas import SubQuery
from src.telemetry import Telemetry

# 三种 query 类型标识（与 M1 复用 core/anchor/discovery；语义按 M3 规范收紧）
INTENT_CORE = "core"
INTENT_ANCHOR = "anchor"
INTENT_DISCOVERY = "discovery"

# M3 规划器版本：所有 15 条 rescue plan 必须用同一 prompt/version 生成；变更时 +1 使旧 hash 失效
M3_PLANNER_VERSION = 1  # M3（已冻结，勿再生成）

# M3.1 Planner Prompt Revision（用户批准的一次且仅一次 prompt 修订）：新唯一版本号
# 目的：Rescue Query 从「自然语言同义改写」进一步变成「互补的学术检索假设」。
M3_1_PLANNER_VERSION = 2

# ≤4 上限：CORE×1 + ANCHOR×1-2 + DISCOVERY×0-1
MAX_CORE = 1
MAX_ANCHOR = 2
MAX_DISCOVERY = 1
MAX_QUERIES = MAX_CORE + MAX_ANCHOR + MAX_DISCOVERY  # 4
PRIORITY = {INTENT_CORE: 5, INTENT_ANCHOR: 4, INTENT_DISCOVERY: 3}

SYSTEM_PROMPT = """你是学术论文检索系统的【稀疏查询救援】规划器。任务：下面这条学术研究问题当前几乎没有可检索的子查询（召回极低）。请为它生成最多 4 条互补的检索 query，让它在学术数据库（OpenAlex）里能召回相关论文。

你只能看到问题文本（一条字符串）。问题可能描述研究方向、方法偏好、对象、约束，甚至提出一个带立场的论点。

输出 JSON 对象：
{"core_queries": ["..."], "anchor_queries": ["..."], "discovery_queries": ["..."]}
每个数组元素是一条检索关键词串（英文，1-10 个词，不用句号）。预算约束：
- core_queries：0-1 条。把复杂问题压缩成一条高密度学术检索表达式（方法+对象，2-6 个词），抓住研究问题主体。
- anchor_queries：1-2 条。从问题里提取【明确给出或可合理推断】的真实实体（方法/模型/基准/数据集/任务名），组成 "<实体> + <任务/方法>" 的检索表达式。优先问题里直接出现或能可靠推断的专名。
- discovery_queries：0-1 条。针对词汇不匹配/组合失效，给 1 条替代学术措辞（同义改写/领域术语），把问题的口语说法翻译成论文标题语言。
- 三个数组加起来 ≤ 4；不必填满，宁缺毋滥。

规则：
- 严禁：不得生成你基于标准答案/Gold 推断出的具体论文标题、作者、DOI、arXiv ID。你只能使用问题文本里明确给出或可合理推断的实体；对不确定真实存在的实体，宁可不写，也不要编造。
- 禁止纯宽泛概念（单独 "LLM"、"language model"、"deep learning"、"neural network"、"machine learning"、"transformer"）。
- 优先让每条 query 像【真实论文标题里会出现的关键词组合】，越具体越好，但必须源自问题本身而非记忆里的特定论文。
- 只输出 JSON 对象，不要 markdown 代码块、不要任何解释。"""

# --------------------------------------------------------------------------
# M3.1 Prompt（唯一一次 Prompt Revision，M3_1_PLANNER_VERSION=2）
# 核心变化：Rescue Query 不再是「同义改写」，而是「互补的学术检索假设」。
# CORE/ANCHOR/DISCOVERY 三者必须承担不同检索假设（不同 benchmark/dataset/method/entity × task
# 组合），禁止把同一概念换词重写。Gold isolation 与 M3 相同：严禁编造具体论文/作者/DOI/arXiv。
# --------------------------------------------------------------------------
SYSTEM_PROMPT_M31 = """你是学术论文检索系统的【稀疏查询救援】规划器。下面这条学术研究问题当前几乎没有可检索的子查询（召回极低）。请为它生成 2-4 条**互补的学术检索假设**，让它在学术数据库（OpenAlex）里能召回不同角度的相关论文。

你只能看到问题文本（一条字符串）。问题可能描述研究方向、方法偏好、对象、约束，甚至提出一个带立场的论点。

输出 JSON 对象：
{"core_queries": ["..."], "anchor_queries": ["..."], "discovery_queries": ["..."]}
每个数组元素是一条检索关键词串（英文，1-10 个词，不用句号）。**总预算 ≤ 4，允许 2-4 条，不必填满；宁缺毋滥。**

三类 query 必须承担【不同】的检索假设，禁止把同一概念换几个词重写：

- core_queries（0-1 条）：把研究问题【压缩】成一条高密度学术名词短语（方法+对象，2-6 词），保留核心 task 与关键 method/domain，去掉自然语言请求成分。
  **禁止**以 "papers about", "research on", "show me", "methods for", "a study of" 开头（这类前缀不含检索信息）。
- anchor_queries（1-2 条）：从问题里识别【明确给出或可合理推断】的 benchmark / dataset / model family / named method / task / domain-specific concept，形成 "<specific anchor> + <task/method/context>" 的组合。
  优先结构：benchmark+task、dataset+method、model family+task、named concept+task、method+domain。
  - 若问题里有明确 benchmark/dataset/model/method 名称，**至少 1 条 ANCHOR 必须包含该实体**。
  - 若问题里没有明确实体：**不得编造实体**，改用具体 task/method concept 组合（仍必须是「专名或具体概念 + 任务」二元素结构）。
- discovery_queries（0-1 条）：专门解决 VOCABULARY_MISMATCH 和 COMPOSITIONAL_QUERY_FAILURE，不是普通 synonym rewrite。
  思考："论文作者描述这个研究问题时，更可能使用什么学术术语或研究范式？"
  允许：学术同义术语、方法类别、社区常用任务名、更标准的研究术语、把原问题两个关键概念重新组合。
  **禁止**：编造具体论文标题/作者/数据集/benchmark/model 名称；只替换一两个普通英文单词形成伪多样性。

规则：
- 严禁：不得生成你基于标准答案/Gold 推断出的具体论文标题、作者、DOI、arXiv ID。你只能使用问题文本里明确给出或可合理推断的实体；对不确定真实存在的实体，宁可不写，也不要编造。
- 禁止纯宽泛概念（单独 "LLM"、"language model"、"deep learning"、"neural network"、"machine learning"、"transformer"）。
- 禁止与原始问题几乎等价的换词改写——每条 query 必须提供一个原始问题未覆盖的检索角度。
- 优先让每条 query 像【真实论文标题里会出现的关键词组合】，越具体越好，但必须源自问题本身而非记忆里的特定论文。
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


# 自然语言请求前缀（M3.1 CORE 规则：禁止 "papers about / research on / show me / methods for /
# do you know / what are..." 开头）。命中即视为无检索信息的 NL 问句 → drop。
_NL_REQUEST_PREFIX = re.compile(
    r"^(papers?(\s+(that|which))?\s+(explore|about|on|using|that\s+explore)|"
    r"research\s+on|show\s+me|methods?\s+for|do\s+you\s+(know|have)|"
    r"what\s+(are|is)|a\s+study\s+of|i\s+(want|need)|give\s+me|"
    r"tell\s+me\s+about|studies?\s+(of|on)|works?\s+(on|about|that)|"
    r"let\s+me\s+know|can\s+you\s+(tell|find|give|show))\b",
    re.IGNORECASE,
)


def _near_dup(tokens: set[str], kept: list[set[str]], threshold: float = 0.7) -> bool:
    """normalized near-dedup：与已保留 query 的 token 集合 Jaccard ≥ threshold 视为重复。"""
    for kt in kept:
        union = len(tokens | kt)
        if union and len(tokens & kt) / union >= threshold:
            return True
    return False


def _near_dup_index(tokens: set[str], kept: list[set[str]], threshold: float = 0.7) -> int | None:
    """返回与 tokens 近义（Jaccard≥threshold）的第一个已保留 query 下标；无则 None。"""
    for i, kt in enumerate(kept):
        union = len(tokens | kt)
        if union and len(tokens & kt) / union >= threshold:
            return i
    return None


def _more_specific(a: set[str], b: set[str]) -> bool:
    """a 是否比 b 更具体：内容词更多则更具体（保留更具体者，避免把宽泛 query 顶掉具体 query）。"""
    return len(a) > len(b)


def _fallback_subqueries(raw_query: str) -> list[SubQuery]:
    """LLM 失败/输出非法时的降级：原始问题（截断）作为唯一 core query。"""
    text = norm_text(raw_query)[:120]
    if not text:
        text = "academic paper search"
    return [SubQuery(id="sq1", query_text=text, intent=INTENT_CORE, priority=PRIORITY[INTENT_CORE])]


class SparsePlanRescue:
    """M3：一次 LLM 调用生成 ≤4 条互补检索 query（core×1 + anchor×1-2 + discovery×0-1）。

    Gold isolation：plan_raw 只接收原始 question 文本，绝不接触 gold。
    失败降级为原始问题（core）。同一 prompt/version 用于全部 rescue query（M3_PLANNER_VERSION）。
    """

    def __init__(self, llm: LLMClient | None = None,
                 version: int = M3_1_PLANNER_VERSION,
                 system_prompt: str = SYSTEM_PROMPT_M31):
        self.llm = llm or LLMClient(tier="fast")
        self.version = version
        self.system_prompt = system_prompt

    async def plan_raw(self, raw_query: str, telemetry: Telemetry | None = None) -> list[SubQuery]:
        """question 文本 -> SubQuery 列表（intent ∈ core/anchor/discovery，2~4 条）。

        默认 M3.1（M3_1_PLANNER_VERSION=2 + SYSTEM_PROMPT_M31）。M3 冻结 plans 仍留在磁盘，不受影响。
        生成后 self.last_raw_counts 记录 LLM 原始输出各 intent 数量（供 generated vs executed/pruned 统计）。
        """
        try:
            data = await self.llm.complete_json(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": raw_query.strip()},
                ],
                max_tokens=1024,
                telemetry=telemetry,
                note="plan_rescue",
            )
            self.last_raw_counts = {t: len(data.get(f"{t}_queries") or []) if isinstance(data.get(f"{t}_queries"), list) else 0
                                    for t in (INTENT_CORE, INTENT_ANCHOR, INTENT_DISCOVERY)}
            subs = self._build_subs(data, original_text=raw_query)
            if subs:
                return subs
            raise LLMError("rescue planner 返回空 query 列表")
        except (LLMError, ValidationError) as e:
            if telemetry:
                telemetry.add_fallback(f"plan_rescue: {type(e).__name__}")
            return _fallback_subqueries(raw_query)

    def _build_subs(self, data: dict, original_text: str | None = None) -> list[SubQuery]:
        """{core,anchor,discovery} -> 过滤/去重 -> SubQuery 列表（2~4，core×1+anchor×1-2+disc×0-1）。

        Diversity Gate（M3.1 确定性最小过滤，无 Quality Selector）：
        - exact dup → drop；normalized dup → drop。
        - 与 original 几乎等价 → drop（避免同义改写，必须提供新检索角度）。
        - 两个 rescue query 核心 token 高度重合且检索假设相同 → 保留更具体者。
        - 明显只有宽泛领域词、没有 task/method/entity 信息 → drop。
        """
        caps = {INTENT_CORE: MAX_CORE, INTENT_ANCHOR: MAX_ANCHOR, INTENT_DISCOVERY: MAX_DISCOVERY}
        picked: list[tuple[int, str, str]] = []  # (priority, type, text)
        seen: set[str] = set()                   # normalized exact-dedup
        kept_tokens: list[set[str]] = []         # near-dedup 参考（与 picked 同序）
        orig_toks = _content_tokens(norm_text(original_text)) if original_text else set()
        for qtype in (INTENT_CORE, INTENT_ANCHOR, INTENT_DISCOVERY):
            items = data.get(f"{qtype}_queries") or []
            if not isinstance(items, list):
                continue
            for item in items:
                if len(picked) >= MAX_QUERIES:
                    break
                if sum(1 for _, t, _ in picked if t == qtype) >= caps[qtype]:
                    break
                if not isinstance(item, str):
                    continue
                text = " ".join(item.split())
                key = norm_text(text)
                if not key or key in seen:
                    continue
                toks = _content_tokens(key)
                # 泛化/单概念/NL 请求前缀/与 original 几乎等价 → drop（不凑数）
                if not toks or _is_too_generic(key) or _NL_REQUEST_PREFIX.match(key) \
                        or _near_dup(toks, [orig_toks]):
                    continue
                dup_j = _near_dup_index(toks, kept_tokens)
                if dup_j is not None:
                    # 两个 rescue query 高度重合且检索假设相同：保留更具体者
                    if _more_specific(toks, kept_tokens[dup_j]):
                        kept_tokens[dup_j] = toks
                        picked[dup_j] = (PRIORITY[qtype], qtype, text)
                        seen.add(key)
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
