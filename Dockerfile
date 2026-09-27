# ScholarTrace 竞赛镜像：API 服务 + 评测体系全量。
# 镜像内含 src/ s1/ api/ config/ configs/ scripts/ tests/ eval/ data/ frontend/ ——
# 既跑 HTTP 服务，也能在容器里直接跑离线评测与单测（无网络、无 LLM）。
#
# 基础镜像按 digest 固定（而非 :3.14-slim 标签）：标签会随时间指向新构建，
# digest 不会。这是「同 commit 两次构建产物一致」的前提之一，
# 另一半是 requirements.lock 里锁定的依赖版本。
FROM python:3.14-slim@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ARG GIT_SHA=unknown
LABEL org.opencontainers.image.title="scholartrace-contest" \
      org.opencontainers.image.description="ScholarTrace Contest API + 评测体系" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/Lucas-cs11/scholartrace"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# 依赖层单独成层：requirements.lock 不变即命中缓存，代码改动不必重装依赖。
#
# --no-compile + 自行 compileall 是为了可重复构建，不是省事：
# pip 默认在安装时编译 .pyc，而 .pyc 头部会写入源码文件的 mtime（构建时刻）。
# 于是两次构建的 .pyc 内容不同 —— 976 个文件，整个层哈希就变了。
# 换成 unchecked-hash 模式后，.pyc 只由源码字节决定，与时间无关；
# 预编译保留下来，容器启动就不必每次重新编译字节码。
COPY requirements.lock ./
RUN pip install --no-cache-dir --no-compile -r requirements.lock \
 && python -m compileall -q --invalidation-mode unchecked-hash \
      /usr/local/lib/python3.14/site-packages

# 全量代码树。内容由 .dockerignore 裁定（= git 跟踪内容），故镜像内容由 commit 决定。
COPY . .

# 非 root 运行；data/ 需可写（SQLite 搜索历史），eval/runs 需可写（评测落盘）
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data /app/eval/runs \
 && chown -R app:app /app
USER app

# 容器内固定 8100；三环境「同机不同端口」由 compose 做宿主机端口映射
EXPOSE 8100

# slim 镜像没有 curl，用标准库自检，不为此多装一个包
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3).status == 200 else 1)"

# sh -c + exec：为的是让 LOG_LEVEL 能按环境注入（dev 看得到细节，prod 不刷屏）。
# exec 让 uvicorn 顶替 shell 成为 PID 1，SIGTERM 才能直达，compose stop 才干净。
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8100 --workers 1 --log-level ${LOG_LEVEL:-info}"]
