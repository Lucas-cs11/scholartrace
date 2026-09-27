#!/usr/bin/env bash
# 构建可重复的 ScholarTrace 镜像 —— 这是唯一的镜像构建入口。
#
# 为什么不用 `docker compose build`：
#   compose 的构建路径没有 rewrite-timestamp。缺了它，层 tar 里文件/目录的时间戳
#   就是构建时刻，两次构建的层哈希必然不同。所以 compose 只负责运行，构建归这里。
#
# 为什么需要 rewrite-timestamp：
#   它把层内所有时间戳归一为 SOURCE_DATE_EPOCH。配合 Dockerfile 里的
#   `pip install --no-compile` + hash 模式 compileall（消除 .pyc 里内嵌的源码 mtime），
#   两次构建的产物才逐字节相同。
#
# 用法：
#   scripts/build_image.sh              # 构建，产出 dist/scholartrace-<sha>.tar 并装载
#   scripts/build_image.sh --verify     # 额外再构建一次并比对哈希，验证可重复性
#
# 前置：docker 与 docker buildx 插件（Ubuntu: apt install docker-buildx）。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${ST_IMAGE:-scholartrace-api:latest}"
OUT_DIR="${ST_DIST:-dist}"
VERIFY=0
[[ "${1:-}" == "--verify" ]] && VERIFY=1

command -v docker >/dev/null 2>&1 || { echo "错误：未找到 docker" >&2; exit 1; }
docker buildx version >/dev/null 2>&1 || {
  echo "错误：未找到 docker buildx 插件。Ubuntu 上：sudo apt install docker-buildx" >&2
  exit 1
}

# 同 commit 必然同值 —— 这是产物与构建时刻无关的前提。
GIT_SHA="$(git rev-parse HEAD)"
SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)"
export SOURCE_DATE_EPOCH

mkdir -p "$OUT_DIR"
TARBALL="$OUT_DIR/scholartrace-${GIT_SHA:0:12}.tar"

build_once() {
  local dest="$1"
  rm -f "$dest"
  docker buildx build \
    --build-arg "GIT_SHA=$GIT_SHA" \
    --build-arg "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
    --output "type=docker,name=$IMAGE,rewrite-timestamp=true,dest=$dest" \
    .
}

echo "commit      : $GIT_SHA"
echo "created 固定: $(date -u -d "@$SOURCE_DATE_EPOCH" +%FT%TZ)  (SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH)"
echo "产物        : $TARBALL"
echo

build_once "$TARBALL"
HASH="$(sha256sum "$TARBALL" | cut -d' ' -f1)"
echo
echo "构建完成  sha256=$HASH"

if [[ "$VERIFY" == "1" ]]; then
  echo
  echo "== 验证可重复性：再构建一次并比对 =="
  SECOND="$OUT_DIR/.verify-${GIT_SHA:0:12}.tar"
  build_once "$SECOND"
  HASH2="$(sha256sum "$SECOND" | cut -d' ' -f1)"
  rm -f "$SECOND"
  if [[ "$HASH" == "$HASH2" ]]; then
    echo "两次构建字节一致 ✅  sha256=$HASH"
  else
    echo "两次构建不一致 ❌" >&2
    echo "  第一次 $HASH" >&2
    echo "  第二次 $HASH2" >&2
    exit 1
  fi
fi

echo
echo "装载镜像 …"
docker load -i "$TARBALL" | tail -1
docker image ls "$IMAGE" --format '  {{.Repository}}:{{.Tag}}  {{.ID}}  {{.Size}}'

echo
echo "下一步：docker compose --env-file deploy/env/dev.env up -d"
