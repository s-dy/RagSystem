# syntax=docker/dockerfile:1.4

# 多阶段构建：builder 用于安装并构建虚拟环境，production 仅拷贝运行时环境
FROM docker.m.daocloud.io/library/python:3.11-slim AS builder

WORKDIR /build

# 安装构建依赖（合并为单层以减少镜像层）
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖描述以利用缓存（仅当 pyproject/uv.lock 变化时才重新安装包）
COPY pyproject.toml uv.lock ./

# 安装 uv（使用 pip 缓存挂载以加速重复构建）
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir uv

# 创建虚拟环境
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# 再复制代码并在 venv 中安装（使用 pip 缓存）
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip uv pip install --system -e .

### 生产阶段：只包含运行时与虚拟环境
FROM docker.m.daocloud.io/library/python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    TZ=Asia/Shanghai

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装运行时依赖（合并为单层）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制虚拟环境（来自 builder）并设置 PATH
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# 复制项目文件到最终镜像（.dockerignore 生效后可避免大文件进入镜像）
COPY . .

# 创建必要目录（尽量保持最小）
RUN mkdir -p uploads logs tf_models frontend

# 暴露端口与健康检查
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/system/models || exit 1

CMD ["python", "main.py"]
