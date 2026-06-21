from typing import Tuple, List
from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger
from app.config import config
from app.services.search_service import search_service


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """
    从研报知识库中检索相关信息。
    当需要查找具体数据、行业分析、公司研究等专业内容时使用此工具。

    Args:
        query: 搜索查询，尽量使用具体的关键词

    Returns:
        (格式化上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索: '{query}'")
        docs = search_service.search(query, top_k=config.rag_top_k)

        if not docs:
            return "未找到相关信息。", []

        # 格式化为 LLM 可读的上下文
        parts = []
        for i, doc in enumerate(docs, 1):
            file_name = doc.metadata.get("file_name", "未知来源")
            parts.append(
                f"【参考资料 {i}】来源: {file_name}\n{doc.page_content}\n"
            )

        context = "\n".join(parts)
        logger.info(f"检索完成: {len(docs)} 篇参考资料")
        return context, docs

    except Exception as e:
        logger.error(f"检索异常: {e}")
        return f"检索时发生错误: {str(e)}", []