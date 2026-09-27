#!/bin/bash
# 等待 OpenAlex 配额恢复后自动串行评测：B1 -> B3 -> B4 -> B2。
# B1 首跑全量召回并落盘磁盘缓存；B2/B3/B4 复用缓存（不重复消耗配额）。
set -euo pipefail
cd /home/ubuntu/scholartrace-contest

MAILTO="lujie.cs%40gmail.com"
URL="https://api.openalex.org/works?search=test&per-page=1&mailto=${MAILTO}"

echo "[$(date)] 等待 OpenAlex 配额恢复..."
while true; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 20 "$URL" || echo 000)
  if [ "$code" = "200" ]; then
    echo "[$(date)] 配额已恢复 (HTTP 200)，开始评测"
    break
  fi
  echo "[$(date)] 配额未恢复 (HTTP $code)，60s 后重试..."
  sleep 60
done

PY=/home/ubuntu/scholartrace-contest/.venv/bin/python

echo "[$(date)] ===== B1 全量（召回 + 落盘缓存）====="
"$PY" scripts/run_b1.py --out /tmp/b1_v2.json > /tmp/eval_b1.log 2>&1
echo "[$(date)] B1 完成，见 /tmp/eval_b1.log"

echo "[$(date)] ===== B3 全量（复用缓存 + LLM 精排）====="
"$PY" scripts/run_b3.py --out /tmp/b3_v2.json > /tmp/eval_b3.log 2>&1
echo "[$(date)] B3 完成，见 /tmp/eval_b3.log"

echo "[$(date)] ===== B4 全量（复用缓存 + 缓存/预算）====="
"$PY" scripts/run_b4.py --out /tmp/b4_v2.json > /tmp/eval_b4.log 2>&1
echo "[$(date)] B4 完成，见 /tmp/eval_b4.log"

echo "[$(date)] ===== B2 全量（复用缓存 + OpenCitations 引文）====="
"$PY" scripts/run_b2.py --out /tmp/b2_v2.json > /tmp/eval_b2.log 2>&1
echo "[$(date)] B2 完成，见 /tmp/eval_b2.log"

echo "[$(date)] 全部评测完成"
