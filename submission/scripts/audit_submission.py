"""S2-F: Submission Package Audit.

Validates contest submission package for common security/reproducibility issues:
- No .env files
- No API keys
- No Gold labels in production runtime
- No eval-only code in search pipeline
- No absolute local paths
- Required configs exist
- Tests pass
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def audit_no_env_files() -> tuple[bool, list[str]]:
    """Check for .env files in repo."""
    env_files = list(REPO.rglob(".env*"))
    # Exclude .env.example
    env_files = [f for f in env_files if f.name != ".env.example"]
    return len(env_files) == 0, [str(f.relative_to(REPO)) for f in env_files]


def audit_no_api_keys() -> tuple[bool, list[str]]:
    """Scan for potential API keys in Python files."""
    issues = []
    patterns = [
        re.compile(r'api[_-]?key\s*=\s*["\'][^"\']{20,}["\']', re.IGNORECASE),
        re.compile(r'sk-[a-zA-Z0-9]{20,}'),  # OpenAI-style keys
        re.compile(r'token\s*=\s*["\'][^"\']{32,}["\']', re.IGNORECASE),
    ]

    for py_file in REPO.rglob("*.py"):
        if ".venv" in str(py_file) or "venv" in str(py_file):
            continue
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern.search(content):
                issues.append(str(py_file.relative_to(REPO)))
                break

    return len(issues) == 0, issues


def audit_no_gold_in_production() -> tuple[bool, list[str]]:
    """Check that production search code doesn't import Gold eval."""
    issues = []

    # Production modules should not import eval.harness or eval.gold
    production_modules = [
        REPO / "s1/pipeline.py",
        REPO / "src/engine.py",
        REPO / "src/ranker.py",
        REPO / "src/planner.py",
    ]

    forbidden = ["eval.harness", "eval.gold", "match_gold"]

    for module in production_modules:
        if not module.exists():
            continue
        content = module.read_text(encoding="utf-8")
        for forbidden_import in forbidden:
            if forbidden_import in content:
                issues.append(f"{module.relative_to(REPO)}: imports {forbidden_import}")

    return len(issues) == 0, issues


def audit_no_absolute_paths() -> tuple[bool, list[str]]:
    """Scan for hardcoded absolute paths."""
    issues = []
    pattern = re.compile(r'["\']/(Users|home|mnt|var|tmp)/[^"\']+["\']')

    for py_file in REPO.rglob("*.py"):
        if ".venv" in str(py_file) or "venv" in str(py_file):
            continue
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        matches = pattern.findall(content)
        if matches:
            # Exclude /tmp/ for legitimate temp files
            filtered = [m for m in matches if not m.startswith('"/tmp/') and not m.startswith("'/tmp/")]
            if filtered:
                issues.append(f"{py_file.relative_to(REPO)}: {filtered[:3]}")

    return len(issues) == 0, issues


def audit_configs_exist() -> tuple[bool, list[str]]:
    """Check required config files exist."""
    required = [
        REPO / "configs/fast.yaml",
        REPO / "configs/deep.yaml",
    ]
    missing = [str(f.relative_to(REPO)) for f in required if not f.exists()]
    return len(missing) == 0, missing


def audit_readme_exists() -> tuple[bool, list[str]]:
    """Check README exists."""
    readme = REPO / "README.md"
    return readme.exists(), [] if readme.exists() else ["README.md"]


def audit_schemas_exist() -> tuple[bool, list[str]]:
    """Check structured output and SearchTrace schemas exist."""
    schema_file = REPO / "s1/schemas.py"
    if not schema_file.exists():
        return False, ["s1/schemas.py missing"]

    content = schema_file.read_text(encoding="utf-8")
    required_classes = ["StructuredResult", "S1SearchTrace"]
    missing = [cls for cls in required_classes if cls not in content]

    return len(missing) == 0, missing


def audit_corpus_fingerprint() -> tuple[bool, list[str]]:
    """Check corpus version constant exists."""
    eval_script = REPO / "scripts/run_eval.py"
    if not eval_script.exists():
        return False, ["scripts/run_eval.py missing"]

    content = eval_script.read_text(encoding="utf-8")
    has_version = "EVAL_CORPUS_VERSION" in content
    has_pasa = "pasa_realscholar_test" in content

    if not (has_version and has_pasa):
        return False, ["EVAL_CORPUS_VERSION not properly defined"]

    return True, []


def audit_tests_pass() -> tuple[bool, list[str]]:
    """Run test suite."""
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/", "-q", "--tb=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
        passed = result.returncode == 0
        errors = [] if passed else [f"Tests failed: {result.stdout[-200:]}"]
        return passed, errors
    except Exception as e:
        return False, [f"Test execution error: {e}"]


def main():
    print("[S2-F] Submission Package Audit")
    print("=" * 60)

    audits = [
        ("No .env files", audit_no_env_files),
        ("No hardcoded API keys", audit_no_api_keys),
        ("No Gold in production imports", audit_no_gold_in_production),
        ("No absolute local paths", audit_no_absolute_paths),
        ("Required configs exist", audit_configs_exist),
        ("README exists", audit_readme_exists),
        ("Schemas exist", audit_schemas_exist),
        ("Corpus fingerprint exists", audit_corpus_fingerprint),
        ("Tests pass", audit_tests_pass),
    ]

    results = {}
    all_pass = True

    for name, audit_fn in audits:
        passed, issues = audit_fn()
        results[name] = {"passed": passed, "issues": issues}
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
        if issues:
            for issue in issues[:5]:  # Limit output
                print(f"           - {issue}")
        all_pass = all_pass and passed

    # Write JSON report
    s2_dir = REPO / "eval/runs/s2"
    s2_dir.mkdir(parents=True, exist_ok=True)
    out = s2_dir / "s2_submission_audit.json"
    report = {
        "overall": "PASS" if all_pass else "FAIL",
        "audits": results,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"[S2-F] Audit report → {out}")
    if all_pass:
        print("[S2-F] ✅ SUBMISSION_AUDIT = PASS")
        return 0
    else:
        print("[S2-F] ❌ SUBMISSION_AUDIT = FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
