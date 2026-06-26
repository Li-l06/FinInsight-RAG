FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# 先只复制依赖文件（利用 Docker 缓存层）
COPY pyproject.toml uv.lock ./

# 安装依赖
RUN uv sync --frozen --no-dev

# 复制所有源码
COPY . .

# ChromaDB 数据持久化目录（docker-compose 挂载卷覆盖）
RUN mkdir -p /app/chroma_db /app/uploads

EXPOSE 9900

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]