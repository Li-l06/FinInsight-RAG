from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger
from app.config import config
from app.services.embedding_service import embedding_service
import chromadb

class VectorStoreService:
    """ChromaDB 向量存储管理"""

    def __init__(self):
        self._store: Optional[Chroma] = None

    def initialize(self) -> Chroma:
        """初始化向量存储"""
        if config.chroma_host:
            # Docker 模式：连接独立的 ChromaDB 容器
            client = chromadb.HttpClient(
                host=config.chroma_host,
                port=config.chroma_port,
            )
            self._store = Chroma(
                client=client,
                collection_name=config.chroma_collection_name,
                embedding_function=embedding_service,
            )
            logger.info(
                f"ChromaDB 就绪 (HTTP): {config.chroma_host}:{config.chroma_port}, "
                f"collection={config.chroma_collection_name}"
            )
        else:
            # 本地模式：内嵌 PersistentClient
            self._store = Chroma(
                collection_name=config.chroma_collection_name,
                embedding_function=embedding_service,
                persist_directory=config.chroma_persist_dir,
            )
            logger.info(
                f"ChromaDB 就绪 (本地): {config.chroma_persist_dir}, "
                f"collection={config.chroma_collection_name}"
            )
        return self._store

    @property
    def store(self) -> Chroma:
        if self._store is None:
            raise RuntimeError("Vector store 未初始化，请先调用 initialize()")
        return self._store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档到向量库"""
        if not documents:
            return []
        ids = self.store.add_documents(documents)
        logger.info(f"向量库已添加 {len(documents)} 个文档块")
        return ids

    def as_retriever(self, top_k: int = 5, **kwargs) -> Any:
        """获取 LangChain Retriever 接口"""
        return self.store.as_retriever(
            search_kwargs={"k": top_k, **kwargs}
        )

    def delete_by_source(self, source: str) -> None:
        """按文件来源删除向量"""
        self.store.delete(filter={"source": source})
        logger.info(f"已删除 source={source} 的向量数据")

    def similarity_search_with_score(
        self, query: str, k: int = 15
    ) -> List[tuple]:
        """向量相似度搜索，返回 (Document, score)"""
        return self.store.similarity_search_with_relevance_scores(query, k=k)

    def get_all_documents(self) -> Dict[str, List[str]]:
        """获取所有文档的来源文件列表"""
        data = self.store.get()
        sources = set()
        if data and data.get("metadatas"):
            for meta in data["metadatas"]:
                src = meta.get("source", "unknown")
                sources.add(src)
        return {"documents": sorted(sources), "total_chunks": len(data["ids"]) if data and data.get("ids") else 0}


vector_store_service = VectorStoreService()