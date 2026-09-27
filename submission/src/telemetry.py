"""统一成本/延时日志（telemetry）。

竞赛用于效率分，产品用于单位经济。每次检索动作、LLM 调用都必须记账。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class Telemetry:
    api_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    fallback_count: int = 0
    api_latency_ms: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def add_api(self, api: str, latency_ms: float, tokens: int = 0, note: str = "") -> None:
        self.api_calls += 1
        self.api_latency_ms.append(latency_ms)
        self.events.append(
            {
                "kind": "api",
                "api": api,
                "latency_ms": round(latency_ms, 1),
                "tokens": tokens,
                "note": note,
                "ts": round(time.time(), 3),
            }
        )

    def add_llm(self, model: str, latency_ms: float, in_tokens: int, out_tokens: int, note: str = "") -> None:
        self.llm_calls += 1
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens
        self.events.append(
            {
                "kind": "llm",
                "model": model,
                "latency_ms": round(latency_ms, 1),
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "note": note,
                "ts": round(time.time(), 3),
            }
        )

    def add_cache_hit(self, note: str = "") -> None:
        self.cache_hits += 1
        self.events.append({"kind": "cache_hit", "note": note, "ts": round(time.time(), 3)})

    def add_fallback(self, note: str = "") -> None:
        self.fallback_count += 1
        self.events.append({"kind": "fallback", "note": note, "ts": round(time.time(), 3)})

    @property
    def total_latency_ms(self) -> float:
        return round(sum(self.api_latency_ms), 1)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> dict:
        return {
            "api_calls": self.api_calls,
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "fallback_count": self.fallback_count,
            "total_api_latency_ms": self.total_latency_ms,
        }

    def dump(self) -> str:
        return json.dumps(self.summary(), ensure_ascii=False)
