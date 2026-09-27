"""测试全局夹具。

tenacity 的重试策略里 wait_exponential(min=5, max=60) 是为线上限流设计的；
在测试中触发重试路径（如适配器 500 分支）会真等约 255 秒，CI 不可用。
这里把 asyncio.sleep 换成「只让出一次调度、不真正等待」，
重试次数与分支逻辑不变，仅去掉墙钟等待。
"""
from __future__ import annotations

import asyncio

import pytest

_real_sleep = asyncio.sleep


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    async def _instant(seconds, result=None):
        return await _real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _instant)
