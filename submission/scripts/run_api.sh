#!/bin/bash
# ScholarTrace Contest API 启动脚本（uvicorn）。
# 用法：./scripts/run_api.sh [--port 8100] [--workers 1]
set -euo pipefail
cd "$(dirname "$0")/.."

PORT=8100
WORKERS=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

exec .venv/bin/python -m uvicorn api.main:app \
  --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
