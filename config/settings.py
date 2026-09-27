"""全局配置（pydantic-settings，从 .env 注入）。"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM（B0 不需要；parser/rerank 需要时再填）----
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = ""            # 深度重排/复杂步骤用强模型
    llm_fast_model: str = ""       # 解析/轻摘要用低成本模型

    # ---- 学术搜索 API ----
    semantic_scholar_api_key: str = ""  # 可选，提高 S2 限流
    openalex_mailto: str = ""           # 可选，进入 polite pool
    # 主召回源：openalex（默认，配额受限）/ crossref（免费无配额）/ s2（无 key 1 req/s）
    recall_source: str = "openalex"
    # 多源召回（方向1）：True 时子查询同时走 OpenAlex + S2，合并去重（S2 结果打 source=s2_recall）
    enable_s2_recall: bool = False

    # ---- 预算默认值 ----
    budget_max_api_calls: int = 50
    budget_max_rounds: int = 4
    budget_max_tokens_per_query: int = 20000
    top_k: int = 20

    # ---- 评测 ----
    gold_path: str = "eval/gold/challenges_v1.jsonl"
    runs_dir: str = "eval/runs"


settings = Settings()
