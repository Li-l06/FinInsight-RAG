"""改写结果关键词校验 — 确保 LLM 没改丢核心实体"""

import jieba


def validate_rewrite(original: str, rewritten: str, threshold: float = 0.7) -> bool:
    """改写后必须保留原始 query 中 >= 70% 的关键实体词"""
    if not original or not rewritten:
        return False

    original_words = list(jieba.cut(original))
    # 取长度 > 1 的词作为实体词（过滤"的""了""吗"等虚词）
    entities = {w.lower() for w in original_words if len(w) > 1}

    if not entities:
        return True

    rewritten_text = rewritten.lower()
    matched = sum(1 for e in entities if e in rewritten_text)
    ratio = matched / len(entities)

    if ratio < threshold:
        from loguru import logger
        lost = entities - {e for e in entities if e in rewritten_text}
        logger.debug(f"改写校验不通过，丢失关键词: {lost}")
    return ratio >= threshold