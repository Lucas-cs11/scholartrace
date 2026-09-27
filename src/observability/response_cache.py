"""HTTP 响应缓存（Phase 2 instrumented replay 基础设施）。

模式：
- write:  首次真实请求，把响应按规范化 key 落盘；同一 run 内重复请求直接返回缓存。
- replay: 只读缓存；命中返回，未命中抛 CacheMiss（保证完全离线，绝无网络请求）。

作用边界：
- 只缓存「学术 API 的 HTTP 响应文本」，不改任何检索/排序/精排逻辑。
- 每次实际发起网络请求（含重试）都会在 attempts[logical_key] 上 +1，
  用于区分 logical API call（算法意图，recorder 记录）与 physical HTTP call（含重试）。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Awaitable, Callable

import httpx


class CacheMiss(Exception):
    """replay 模式未命中缓存：说明该请求在 write run 中未发生或缓存缺失。"""


def cache_key(provider: str, endpoint: str, query_or_seed: str) -> str:
    """逻辑调用 key：{provider}:{endpoint}:{query_or_seed}。

    provider 与 search.py 的 recorder 记录一致（openalex/crossref/opencitations/s2）。
    """
    return f"{provider}:{endpoint}:{query_or_seed}"


FetchFn = Callable[[], Awaitable[httpx.Response]]


class ResponseCache:
    """HTTP 响应缓存。mode=write 时联网并落盘，mode=replay 时只读缓存。"""

    def __init__(self, cache_dir: str | Path, mode: str = "write"):
        self.cache_dir = Path(cache_dir)
        self.mode = mode
        if mode not in ("write", "replay"):
            raise ValueError(f"mode 必须为 write/replay，got {mode}")
        self._entries: dict[str, dict] = {}  # sha256(method|url|params) -> {status,text,headers,ts,logical_key}
        self._attempts: dict[str, int] = {}  # logical_key -> 物理 HTTP 尝试次数（含重试）
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        idx = self.cache_dir / "index.jsonl"
        if not idx.exists():
            return
        for line in idx.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._entries[d["h"]] = d
            lk = d.get("logical_key")
            if lk:
                self._attempts[lk] = max(self._attempts.get(lk, 0), int(d.get("attempts", 1)))

    def _flush(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for ent in self._entries.values():
            lines.append(json.dumps(ent, ensure_ascii=False))
        (self.cache_dir / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _hash(method: str, url: str, params: dict | None) -> str:
        canon = json.dumps(sorted((params or {}).items()), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(f"{method}|{url}|{canon}".encode()).hexdigest()[:20]

    # ------------------------------------------------------------------
    def physical_attempts(self, logical_key: str) -> int:
        """某逻辑调用的物理 HTTP 尝试次数（write run 中含重试；replay 为 0）。"""
        return self._attempts.get(logical_key, 0)

    def total_physical_attempts(self) -> int:
        return sum(self._attempts.values())

    def attempt_stats(self) -> dict[str, int]:
        return dict(self._attempts)

    async def get_or_fetch(
        self,
        logical_key: str,
        method: str,
        url: str,
        params: dict | None,
        fetch: FetchFn,
    ) -> httpx.Response:
        """统一 HTTP 入口：write 模式缓存响应，replay 模式只读缓存。

        fetch 为实际网络请求闭包。重试由上层（tenacity）驱动：重试会再次调用
        本方法，attempts 会累计，从而记录真实的物理调用次数。
        """
        self._attempts[logical_key] = self._attempts.get(logical_key, 0) + 1
        h = self._hash(method, url, params)
        ent = self._entries.get(h)
        if ent is not None and ent.get("status", 200) < 400:
            # 只重放成功（2xx/3xx）响应：错误响应（如 429）不缓存、不重放，
            # 否则 write run 中一次限流会把永久 429 写进缓存，导致 replay 反复失败。
            # 重放语义：缓存的 text 已是解码后的响应体，必须剥掉 gzip/br 编码头与
            # content-length，否则 httpx 重建 Response 时会二次解压而失败（DecodingError）。
            headers = dict(ent.get("headers") or {})
            headers.pop("content-encoding", None)
            headers.pop("content-length", None)
            resp = httpx.Response(
                ent["status"],
                text=ent.get("text", ""),
                headers=headers,
                request=httpx.Request(method, url, params=params),
            )
            # 命中缓存不算新的物理网络调用（attempts 已在上面 +1，标记为缓存命中语义）
            return resp
        # 缓存无此成功响应（或命中过期的错误响应）→ 视为 miss
        if self.mode == "replay":
            raise CacheMiss(logical_key)
        resp = await fetch()
        if resp.status_code < 400:
            self._entries[h] = {
                "h": h,
                "logical_key": logical_key,
                "status": resp.status_code,
                "text": resp.text,
                "headers": dict(resp.headers),
                "attempts": self._attempts[logical_key],
                "ts": time.time(),
            }
        return resp

    def flush(self) -> None:
        """把内存缓存落盘（崩溃保护：长 run 每 query 后调用一次）。"""
        if self.mode == "write":
            self._flush()

    def close(self) -> None:
        self.flush()
