from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.config import config
from app.api import chat, document, health
from app.services.vector_store_service import vector_store_service
from app.services.search_service import search_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    logger.info(f"🚀 FinInsight-RAG 启动中...")
    logger.info(f"http://{config.host}:{config.port}")
    logger.info(f"API 文档: http://{config.host}:{config.port}/docs")

    # 初始化 ChromaDB
    vector_store_service.initialize()

    # 构建 BM25 索引
    search_service.build_bm25_index()

    logger.info("✅ 所有服务就绪")

    yield

    logger.info("🛑 服务关闭")


app = FastAPI(
    title="FinInsight-RAG",
    version="1.0.0",
    description="金融研报 RAG 智能分析 Agent",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["对话"])
app.include_router(document.router, prefix="/api/v1", tags=["文档"])
app.include_router(health.router, prefix="/api/v1", tags=["系统"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )