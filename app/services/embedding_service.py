from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from loguru import logger
from app.config import config


class DashScopeEmbeddings(Embeddings):
    """DashScope Text Embedding — OpenAI 兼容协议"""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = config.dashscope_embedding_model
        self.dimensions = 1024
        logger.info(f"DashScope Embeddings 就绪: model={self.model}, dim={self.dimensions}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        return response.data[0].embedding


embedding_service = DashScopeEmbeddings()