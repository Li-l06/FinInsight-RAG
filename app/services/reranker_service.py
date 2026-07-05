"""Reranker 精排服务 — 调用 DashScope Rerank API"""

from typing import List
import httpx
from langchain_core.documents import Document
from loguru import logger
from app.config import config


class RerankerService:
    """云端 Reranker，对召回文档精排取 Top-N"""

    RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def rerank(
        self, query: str, docs: List[Document], top_n: int | None = None
    ) -> List[Document]:
        """精排并返回 Top-N 文档"""
        if not docs or not config.enable_rerank:
            return docs

        top_n = top_n or config.rerank_top_n
        if len(docs) <= top_n:
            return docs

        try:
            payload = {
                "model": config.rerank_model,
                "query": query,
                "documents": [doc.page_content for doc in docs],
                "parameters": {"top_n": top_n},
            }
            headers = {
                "Authorization": f"Bearer {config.dashscope_api_key}",
                "Content-Type": "application/json",
            }

            with httpx.Client(timeout=15) as client:
                resp = client.post(self.RERANK_URL, json=payload, headers=headers)
                resp.raise_for_status()
                result = resp.json()

            ranked_indices = [item["index"] for item in result["output"]["results"]]
            ranked_docs = [docs[i] for i in ranked_indices]
            logger.info(f"Reranker 精排: {len(docs)} → {len(ranked_docs)} 条")
            return ranked_docs

        except Exception as e:
            logger.warning(f"Reranker 调用失败，降级使用原始结果: {e}")
            return docs[:top_n]


reranker_service = RerankerService()