"""S1 Gold 泄漏自动化校验。

两道防线：
1. 静态：扫描 s1/ 运行期模块源码，禁止出现 gold 相关符号/路径
   （match_gold / eval.gold / challenges_v / load_challenges / 直接读 gold 文件）。
2. 运行时：在离线 search 执行期间挂 open() 审计钩子，断言未打开任何 eval/gold 或
   challenges 文件。frozen plan/cache 属于允许白名单。

用法：
    from s1.leakage import check_static, check_runtime_no_gold
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType

REPO = Path(__file__).resolve().parent.parent
S1_DIR = REPO / "s1"

# 禁止在运行期检索代码中出现的 gold 符号 / 路径片段
FORBIDDEN_SYMBOLS = [
    "match_gold", "load_challenges", "gold_path", "challenges_v",
    "eval/gold", "gold_groups", "compute_p_r_f1",
]
# 离线 search 允许打开的白名单（相对 REPO）
ALLOWED_OPEN = {
    "eval/runs/m3r_append/m3r_query_plans.jsonl",
    "eval/cache/m3r_append/recall_cache.jsonl",
    "eval/runs/m5a_two_round/m5a_round2_plans.jsonl",
    "eval/runs/m5a_two_round/m5a_round2_recall_cache.jsonl",
    "eval/runs/m5a_two_round/m5a_plan_meta.json",
}


class GoldLeakError(RuntimeError):
    """Gold 泄漏校验失败。"""


def check_static() -> list[str]:
    """返回违规清单（空列表 = 通过）。仅扫描 s1/ 运行期代码，跳过 tests/ 与校验器自身。"""
    violations: list[str] = []
    for py in sorted(S1_DIR.glob("*.py")):
        if py.name.startswith("test_") or py.name == "leakage.py":
            continue
        src = py.read_text(encoding="utf-8")
        for sym in FORBIDDEN_SYMBOLS:
            if sym in src:
                violations.append(f"{py.name}: contains forbidden symbol {sym!r}")
    return violations


class _OpenAuditor:
    """包装 builtins.open 记录运行期打开的文件（Python 无 removeaudithook，故用 open-wrap）。"""

    def __init__(self) -> None:
        self.opened: list[str] = []
        self._orig_open = None

    def __enter__(self) -> "_OpenAuditor":
        import builtins
        self._orig_open = builtins.open

        def guarded(file, mode="r", *a, **k):
            try:
                path = Path(file).resolve()
                if not path.is_dir():
                    self.opened.append(str(path))
            except Exception:  # noqa: BLE001
                pass
            return self._orig_open(file, mode, *a, **k)

        self._guarded = guarded
        builtins.open = guarded
        return self

    def __exit__(self, _et: type, _e: BaseException | None, _tb: TracebackType | None) -> None:
        if self._orig_open is not None:
            import builtins
            builtins.open = self._orig_open


def check_runtime_no_gold(search_call) -> list[str]:
    """执行 search_call，校验运行期间未打开任何 gold 文件。返回违规清单。"""
    with _OpenAuditor() as auditor:
        search_call()
    gold_files = [p for p in auditor.opened
                  if ("eval/gold" in p) or ("challenges" in p)]
    return gold_files


def assert_no_gold_leak(search_call) -> None:
    """运行 search_call 并断言无 Gold 泄漏；失败抛 GoldLeakError。"""
    static = check_static()
    if static:
        raise GoldLeakError("静态泄漏：\n  " + "\n  ".join(static))
    runtime = check_runtime_no_gold(search_call)
    if runtime:
        raise GoldLeakError("运行时泄漏（search 打开了 gold 文件）：\n  " + "\n  ".join(runtime))
