#!/bin/bash
# 一键复现全部评测：B0 -> B1 -> B3 -> B4 -> B2 -> FULL。
# B1 首跑全量召回并落盘磁盘缓存；B2/B3/B4/FULL 复用（只耗 LLM/引文）。
# 需 OpenAlex 配额（~1000 credits/天）。摘要写入 /tmp/b{0,1,2,3,4}.json、/tmp/full.json。
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
RUNS=eval/runs
CACHE="$RUNS/_recall_cache.jsonl"
PLAN="$RUNS/_plan_cache.jsonl"

mkdir -p "$RUNS"

echo "===== [$(date)] B0 基线（词法）====="
"$PY" scripts/run_b0.py --out /tmp/b0.json

echo "===== [$(date)] B1 查询理解+子查询召回（首跑，填缓存）====="
"$PY" scripts/run_b1.py --cache "$CACHE" --plan-cache "$PLAN" --out /tmp/b1.json

echo "===== [$(date)] B3 LLM 精排（复用缓存）====="
"$PY" scripts/run_b3.py --cache "$CACHE" --plan-cache "$PLAN" --out /tmp/b3.json

echo "===== [$(date)] B4 缓存+预算（复用缓存）====="
"$PY" scripts/run_b4.py --cache "$CACHE" --plan-cache "$PLAN" --out /tmp/b4.json

echo "===== [$(date)] B2 引文扩展（复用缓存 + OpenCitations）====="
"$PY" scripts/run_b2.py --cache "$CACHE" --plan-cache "$PLAN" --out /tmp/b2.json

echo "===== [$(date)] FULL 全链路（复用缓存）====="
"$PY" scripts/run_full.py --cache "$CACHE" --plan-cache "$PLAN" --out /tmp/full.json

echo "[$(date)] 全部评测完成，摘要见 /tmp/b{0,1,2,3,4}.json /tmp/full.json"
