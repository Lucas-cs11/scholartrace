#!/bin/bash
# M1_R2（ONE_PLANNER_REVISION）：等 OpenAlex 配额恢复（用户换 IP），恢复后自动启动 R2 检索。
cd /Users/lucas/Documents/赛题/scholartrace-contest-backup
LOG=eval/runs/m1_anchor_augmented_r2/watcher.log
echo "$(date '+%H:%M:%S') watcher(R2) 启动，等待配额恢复..." >> "$LOG"
for i in $(seq 1 240); do
  HDR=$(curl -s -D - -o /dev/null "https://api.openalex.org/works?search=test&per-page=1")
  R=$(echo "$HDR" | tr -d '\r' | grep '^x-ratelimit-remaining:' | awk '{print $2}')
  RESET=$(echo "$HDR" | tr -d '\r' | grep '^x-ratelimit-reset:' | awk '{print $2}')
  if [ -n "$R" ] && [ "$R" -ge 100 ] 2>/dev/null; then
    echo "$(date '+%H:%M:%S') 配额恢复 remaining=$R（IP 已切换），启动 M1_R2 检索" >> "$LOG"
    nohup python3 -u scripts/run_m1_anchor.py search --rev 2 >> eval/runs/m1_anchor_augmented_r2/search.log 2>&1 &
    echo "$(date '+%H:%M:%S') search(R2) launched PID $!" >> "$LOG"
    exit 0
  fi
  if [ $((i % 4)) -eq 0 ]; then
    echo "$(date '+%H:%M:%S') 检查 $i: remaining=$R reset=${RESET}s，继续等待" >> "$LOG"
  fi
  sleep 30
done
echo "$(date '+%H:%M:%S') watcher(R2) 超时退出（2 小时内未恢复）" >> "$LOG"
