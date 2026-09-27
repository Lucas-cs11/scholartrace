#!/bin/bash
# M2_QUERY_DENSITY：等 OpenAlex 配额恢复（用户换 IP），恢复后自动启动 M2 检索。
cd /Users/lucas/Documents/赛题/scholartrace-contest-backup
LOG=eval/runs/m2_query_density/watcher.log
echo "$(date '+%H:%M:%S') watcher(M2) 启动，等待配额恢复..." >> "$LOG"
for i in $(seq 1 240); do
  HDR=$(curl -s -D - -o /dev/null "https://api.openalex.org/works?search=test&per-page=1")
  R=$(echo "$HDR" | tr -d '\r' | grep '^x-ratelimit-remaining:' | awk '{print $2}')
  RESET=$(echo "$HDR" | tr -d '\r' | grep '^x-ratelimit-reset:' | awk '{print $2}')
  if [ -n "$R" ] && [ "$R" -ge 200 ] 2>/dev/null; then
    echo "$(date '+%H:%M:%S') 配额恢复 remaining=$R（IP 已切换），启动 M2 检索" >> "$LOG"
    nohup python3 -u scripts/run_m2_density.py search >> eval/runs/m2_query_density/search.log 2>&1 &
    echo "$(date '+%H:%M:%S') search(M2) launched PID $!" >> "$LOG"
    exit 0
  fi
  if [ $((i % 4)) -eq 0 ]; then
    echo "$(date '+%H:%M:%S') 检查 $i: remaining=$R reset=${RESET}s，继续等待" >> "$LOG"
  fi
  sleep 30
done
echo "$(date '+%H:%M:%S') watcher(M2) 超时退出（2 小时内未恢复）" >> "$LOG"
