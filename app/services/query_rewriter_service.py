"""查询改写服务 — 用 LLM 将口语化问题改写为检索友好关键词"""

from app.config import config
from app.core.llm_factory import create_chat_model
from loguru import logger


REWRITE_PROMPT = """你是一个搜索查询优化器。将用户口语化问题改写为适合向量检索的关键词短语。

规则：
1. 保留核心实体名词（公司名、行业名、数据指标名）
2. 保留数字和时间（Q3、2024年、毛利率、15%）
3. 去掉口语词（"帮我查一下""有没有""是什么"）
4. 输出纯关键词，不要加任何解释

示例：
输入：Q3 毛利率环比提升的消费类公司有哪些
输出：消费类公司 Q3 毛利率 环比提升

输入：帮我查一下新能源板块最近的走势
输出：新能源板块 近期走势"""


class QueryRewriterService:
    """LLM 查询改写"""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = create_chat_model(streaming=False)
        return self._model

    def rewrite(self, query: str) -> str:
        """改写查询，异常时返回原始 query"""
        if not config.enable_query_rewrite:
            return query

        try:
            response = self.model.invoke([
                ("system", REWRITE_PROMPT),
                ("user", query),
            ])
            rewritten = response.content.strip()
            if rewritten and rewritten != query:
                logger.debug(f"查询改写: '{query}' → '{rewritten}'")
                return rewritten
            return query
        except Exception as e:
            logger.warning(f"查询改写失败，使用原始查询: {e}")
            return query


query_rewriter_service = QueryRewriterService()