"""
离线检索评估脚本

用法：准备好测试集后，直接 python tests/test_evaluation.py
"""


import asyncio
from app.config import config as cfg
from app.services.search_service import search_service
from app.services.vector_store_service import vector_store_service
from app.services.rag_agent_service import rag_agent_service


# ─── 测试数据（自己标注30条即可） ─────────────────────
# 格式：{"query": "问题", "relevant_docs": ["文件名1", "文件名2"]}
TEST_QUERIES = [
    {
        "query": "2024年消费电子行业营收增速",
        "relevant_docs": ["消费电子行业2024年度策略.pdf"],
    },
    # ... 补充更多
]


async def evaluate_retrieval(k: int = 5):
    """检索指标对比：三组实验"""
    from app.services.reranker_service import reranker_service

    print(f"\n{'='*60}")
    print(f"检索评估对比 (Top-{k})")
    print(f"{'='*60}")

    for mode_name, use_rerank in [("Dense+BM25", False), ("Dense+BM25+Reranker", True)]:
        cfg.enable_rerank = use_rerank
        recall_sum = 0.0
        mrr_sum = 0.0
        total = len(TEST_QUERIES)

        for item in TEST_QUERIES:
            docs = search_service.search(item["query"], top_k=k)
            retrieved_files = [d.metadata.get("file_name", "") for d in docs]

            hits = sum(1 for f in retrieved_files if f in item["relevant_docs"])
            recall_sum += hits / len(item["relevant_docs"]) if item["relevant_docs"] else 0

            for rank, f in enumerate(retrieved_files, 1):
                if f in item["relevant_docs"]:
                    mrr_sum += 1 / rank
                    break

        print(f"\n{mode_name}:")
        print(f"  Recall@{k}: {recall_sum/total:.2%}")
        print(f"  MRR: {mrr_sum/total:.4f}")

    cfg.enable_rerank = True  # 恢复默认


async def evaluate_generation():
    """生成评估：抽样 5 条做人工检查"""
    print("=== 生成质量抽样检查 ===")
    samples = TEST_QUERIES[:5]

    for item in samples:
        answer = await rag_agent_service.query(item["query"], session_id="eval")
        print(f"\n问题: {item['query']}")
        print(f"回答: {answer[:200]}...")
        print(f"期望来源: {item['relevant_docs']}")
        print("---")
        rating = input("评分 (1-5, 相关性/准确性): ")
        print(f"  评分: {rating}")


async def main():
    vector_store_service.initialize()
    search_service.build_bm25_index()
    await evaluate_retrieval(k=5)
    # await evaluate_generation()  # 需要人工参与时取消注释


if __name__ == "__main__":
    asyncio.run(main())