"""S1 配置加载（configs/fast.yaml | deep.yaml）。"""
from __future__ import annotations

from pathlib import Path

import yaml

from s1.schemas import S1Config


def load_config(path: str | Path) -> S1Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    mode = str(raw.get("mode", "fast")).lower()
    # 允许 YAML 里用 snake_case 覆盖默认
    known = {k for k in S1Config.model_fields}
    kwargs = {k: v for k, v in raw.items() if k in known and k != "mode"}
    cfg = S1Config(mode=mode, **kwargs)
    cfg.offline = bool(raw.get("offline", True))
    return cfg
