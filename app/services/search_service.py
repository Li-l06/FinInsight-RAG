from typing import List, Tuple
from rank_bm25 import BM25Okapi
import jieba
from langchain_core.documents import Document
from loguru import logger
from app.config import config
from app.services.vector_store_service import vector_store_service


class SearchService:
    """
    多路召回 + RRF 融合

    路1：Dense 向量检索 (ChromaDB)
    路2：Sparse 关键词检索 (BM25 + jieba 分词)
    融合：Reciprocal Rank Fusion (RRF), k=60
    """

    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: List[Document] = []
        self._bm25_corpus: List[List[str]] = []

    def build_bm25_index(self) -> None:
        """从 ChromaDB 拉取全量文档构建 BM25 索引"""
        data = vector_store_service.store.get()
        if not data or not data.get("documents"):
            logger.warning("ChromaDB 中无数据，BM25 索引为空")
            self._bm25 = None
            return

        self._bm25_docs = []
        self._bm25_corpus = []

        for i, content in enumerate(data["documents"]):
            doc = Document(
                page_content=content,
                metadata=data["metadatas"][i] if data.get("metadatas") else {},
            )
            self._bm25_docs.append(doc)
            # jieba 分词
            tokens = list(jieba.cut(content))
            self._bm25_corpus.append(tokens)

        self._bm25 = BM25Okapi(self._bm25_corpus)
        logger.info(f"BM25 索引构建完成: {len(self._bm25_docs)} 个文档块")

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """多路召回 + 查询改写 + RRF 融合 + Reranker 精排"""
        from app.services.query_rewriter_service import query_rewriter_service
        from app.services.keyword_validator import validate_rewrite
        from app.services.reranker_service import reranker_service

        # 收集所有召回路
        all_docs: List[Document] = []
        queries = [query]  # 原始 query 始终参与

        # 查询改写（双路融合策略：原始 + 改写各自召回，RRF 合并）
        if config.enable_query_rewrite:
            rewritten = query_rewriter_service.rewrite(query)
            if rewritten and rewritten != query and validate_rewrite(query, rewritten):
                queries.append(rewritten)

        # 每条 query 独立召回
        for q in queries:
            dense = self._dense_search(q, top_k=20)
            sparse = self._sparse_search(q, top_k=20)
            all_docs.extend(dense)
            all_docs.extend(sparse)

        # RRF 融合去重
        merged = self._rrf_fusion_all(all_docs, k=60)

        # Reranker 精排
        if config.enable_rerank:
            merged = reranker_service.rerank(query, merged, top_n=config.rerank_top_n)

        return merged[:top_k]

    def _dense_search(self, query: str, top_k: int = 15) -> List[Document]:
        """向量语义检索"""
        try:
            results = vector_store_service.similarity_search_with_score(
                query, k=top_k
            )
            docs = [doc for doc, _score in results]
            logger.debug(f"Dense 检索: {len(docs)} 结果")
            return docs
        except Exception as e:
            logger.error(f"Dense 检索失败: {e}")
            return []

    def _sparse_search(self, query: str, top_k: int = 15) -> List[Document]:
        """BM25 关键词检索"""
        if self._bm25 is None:
            self.build_bm25_index()
        if self._bm25 is None:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)
        # 按分数排序取 Top-K
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        docs = [self._bm25_docs[idx] for idx, _score in ranked if scores[idx] > 0]
        logger.debug(f"BM25 检索: {len(docs)} 结果")
        return docs
    def _rrf_fusion_all(
        self,
        doc_lists: List[Document],  # type: ignore — 实际接受扁平列表，内部用 page_content 去重
        k: int = 60,
    ) -> List[Document]:
        """RRF 融合多个召回结果（按 page_content 去重）"""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for doc in doc_lists:
            key = doc.page_content
            doc_map[key] = doc

        # 对每个唯一文档，按在各路中的最高排名计分
        # 简化：按首次出现顺序赋排名
        seen: dict[str, int] = {}
        for doc in doc_lists:
            key = doc.page_content
            if key not in seen:
                rank = len(seen) + 1
                seen[key] = rank
                scores[key] = scores.get(key, 0) + 1 / (k + rank)

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        logger.info(f"RRF 融合: {len(doc_lists)} 条 → {len(sorted_keys)} 去重结果")
        return [doc_map[k] for k in sorted_keys]

    def _rrf_fusion(
        self,
        list_a: List[Document],
        list_b: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion

        score(d) = Σ 1 / (k + rank_i(d))

        思路：文档在任一榜单中排名越高，融合分数越高
        """
        scores: dict = {}
        doc_map: dict = {}

        # 处理路1
        for rank, doc in enumerate(list_a, 1):
            key = doc.page_content  # 用内容做去重键
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            doc_map[key] = doc

        # 处理路2
        for rank, doc in enumerate(list_b, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            doc_map[key] = doc

        # 排序
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        logger.info(f"RRF 融合: {len(list_a)} + {len(list_b)} → {len(sorted_keys)} 去重结果")
        return [doc_map[key] for key in sorted_keys]


search_service = SearchService()