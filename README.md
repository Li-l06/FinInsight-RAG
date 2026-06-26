# FinInsight-RAG

> 金融研报 RAG 智能分析 Agent — 语义检索 + 多轮对话

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-brightgreen)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

针对券商研究员在海量研报中检索效率低、传统关键词搜索无法理解语义查询（如"Q3 毛利率环比提升的消费类公司"）的痛点，搭建金融研报 RAG 智能问答系统，将单次研报信息查找从 **10 分钟级降至秒级**。

---

## 核心特性

- **双路混合检索** — Dense 向量语义 + BM25 关键词双路并行，RRF 融合去重，弥补单一检索的语义漂移
- **ReAct Agent 对话** — 基于 LangGraph 的多轮对话，自动工具调用（知识检索），SSE 流式输出
- **智能文档切分** — 段落边界感知滑动窗口 + 表格完整性保护（历经 V1→V3 三次迭代）
- **三层稳定性机制** — 防无限循环截断、消息历史动态裁剪、工具调用指数退避重试
- **离线评估体系** — 30 组金融领域标注查询，Recall@5 稳定 85%+，MRR 0.72

---

## 架构

```mermaid
flowchart LR
    A[用户] -->|HTTP/SSE| B[FastAPI]
    B --> C{SearchService}
    C -->|Dense 向量| D[ChromaDB]
    C -->|Sparse BM25| E[BM25 + jieba]
    C -->|RRF 融合| F[Top-K 结果]
    F --> G[RagAgentService]
    G -->|ReAct 循环| H[DashScope LLM]
    H -->|SSE 流式| A
    B --> I[DocumentParser]
    I -->|pdfplumber| J[文本 + 表格]
    J --> K[DocumentSplitter]
    K --> D