# FinInsight-RAG

> 金融研报 RAG 智能分析 Agent — 混合检索 + Reranker 精排 + 查询改写 + 多轮对话

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-orange)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-purple)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-brightgreen)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

针对券商研究员在海量研报中检索效率低、传统关键词搜索无法理解语义查询（如"Q3 毛利率环比提升的消费类公司"）的痛点，搭建金融研报 RAG 智能问答系统。检索侧采用 **Dense + BM25 双路召回 + Reranker 精排** 的级联架构，搭配 LLM 查询改写提升召回覆盖率，将单次研报信息查找从 **10 分钟级降至秒级**。

---

## 核心特性

- **四阶段检索管线** — 查询改写（LLM）→ 双路混合召回（Dense + BM25）→ RRF 融合去重 → Reranker 精排（gte-rerank），逐级提纯
- **ReAct Agent 对话** — 基于 LangGraph 的多轮对话，自动工具调用（知识检索），SSE 流式输出
- **智能文档切分** — 段落边界感知滑动窗口 + 表格完整性保护（历经 V1→V3 三次迭代）
- **三层稳定性机制** — 防无限循环截断、消息历史动态裁剪、工具调用指数退避重试；查询改写附带关键词校验 + 降级兜底
- **离线评估体系** — 30 组金融领域标注查询，Recall@5 稳定 85%+，MRR 0.72

---

## 检索管线

``
用户提问: "Q3 毛利率环比提升的消费类公司有哪些"
  │
  ├─ 查询改写 (LLM): "消费类公司 Q3 毛利率 环比提升"
  │   └─ 关键词校验: 实体词保留率 >= 70% → 通过
  │
  ├─ 双路并行召回 (原始 + 改写各 20 条)
  │   ├─ Dense 向量检索 (text-embedding-v4)
  │   └─ BM25 关键词检索 (jieba 分词)
  │
  ├─ RRF 融合去重 (k=60)
  │
  ├─ Reranker 精排 (gte-rerank) → Top-3
  │
  └─ 送入 LLM 生成带引用的回答
``

---

## 架构

``mermaid
flowchart LR
    A[用户] -->|HTTP/SSE| B[FastAPI]
    B --> QW[查询改写<br/>QueryRewriter]
    QW -->|原始 + 改写双路| C{混合召回}
    C -->|Dense 向量| D[ChromaDB]
    C -->|Sparse BM25| E[BM25 + jieba]
    C -->|RRF 融合| F[候选文档集]
    F -->|精排| RE[Reranker<br/>gte-rerank]
    RE -->|Top-3| G[RagAgentService]
    G -->|ReAct 循环| H[DashScope LLM]
    H -->|SSE 流式| A
    B --> I[DocumentParser]
    I -->|pdfplumber| J[文本 + 表格]
    J --> K[DocumentSplitter]
    K --> D
``

---

## 项目结构

``
RAG-ll/
├── app/
│   ├── api/              # FastAPI 路由 (chat, document, health)
│   ├── core/             # LLM 工厂 (ChatQwen 集成)
│   ├── models/           # Pydantic 请求/响应模型
│   ├── services/         # 核心服务
│   │   ├── document_parser.py          # 文档解析 (pdfplumber)
│   │   ├── document_splitter.py        # V3 智能切分器
│   │   ├── embedding_service.py        # DashScope Embedding
│   │   ├── keyword_validator.py        # 改写关键词校验
│   │   ├── query_rewriter_service.py   # LLM 查询改写
│   │   ├── rag_agent_service.py        # ReAct Agent (LangGraph)
│   │   ├── reranker_service.py         # Reranker 精排 (gte-rerank)
│   │   ├── search_service.py           # 混合检索 + RRF 融合
│   │   └── vector_store_service.py     # ChromaDB 管理
│   ├── tools/
│   │   └── knowledge_tool.py           # 知识检索 Agent 工具
│   ├── utils/
│   │   └── logger.py                   # Loguru 日志配置
│   ├── config.py         # Pydantic Settings 配置管理
│   └── main.py           # FastAPI 入口
├── tests/
│   └── test_evaluation.py              # 离线评估 (Recall@5, MRR)
├── chroma_db/            # ChromaDB 持久化目录
├── uploads/              # 上传文件临时目录
├── docker-compose.yml    # Docker 编排
├── Dockerfile            # Docker 镜像
├── pyproject.toml        # 项目依赖
└── README.md
``

---

## 快速开始

### 环境要求
- Python 3.11+
- DashScope API Key ([获取地址](https://dashscope.aliyun.com/))

### 本地运行

``powershell
# 1. 克隆项目
git clone https://github.com/Li-l06/FinInsight-RAG.git
cd FinInsight-RAG

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 3. 配置 API Key
# 编辑 .env，填入你的 DASHSCOPE_API_KEY

# 4. 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 9900

# 5. 访问 API 文档
# http://localhost:9900/docs
``

### Docker 运行

``powershell
docker compose up -d
``

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/v1/health | 健康检查 |
| POST | /api/v1/chat | 同步对话 |
| POST | /api/v1/chat/stream | SSE 流式对话 |
| POST | /api/v1/chat/clear | 清除会话 |
| GET | /api/v1/chat/session/{id} | 获取会话历史 |
| POST | /api/v1/document/upload | 上传文档 |
| POST | /api/v1/document/list | 文档列表 |

### 使用示例

``bash
# 流式对话
curl -X POST "http://localhost:9900/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"id":"test","question":"2024年消费电子行业营收增速"}' \
  --no-buffer

# 上传研报
curl -X POST "http://localhost:9900/api/v1/document/upload" \
  -F "file=@report.pdf"
``

---

## 配置说明

通过 .env 文件配置（完整列表见 .env.example）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| DASHSCOPE_API_KEY | — | DashScope API Key（必填） |
| DASHSCOPE_MODEL | qwen-max | LLM 模型 |
| DASHSCOPE_EMBEDDING_MODEL | 	ext-embedding-v4 | Embedding 模型 |
| CHUNK_MAX_SIZE | 800 | 文档切分窗口大小 |
| CHUNK_OVERLAP | 100 | 切分重叠字符数 |
| RAG_TOP_K | 5 | 最终返回给 LLM 的文档数 |
| ENABLE_RERANK | 	rue | 启用 Reranker 精排 |
| RERANK_MODEL | gte-rerank | Reranker 模型 |
| RERANK_TOP_N | 3 | 精排后保留条数 |
| ENABLE_QUERY_REWRITE | 	rue | 启用查询改写 |

---

## 离线评估

``powershell
# 需要先准备好 TEST_QUERIES 标注数据
python tests/test_evaluation.py
``

输出三组对比：
- Dense+BM25（基线）
- Dense+BM25+Reranker（精排增强）

指标：Recall@5、MRR

---

## 参考资源

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [DashScope 文档](https://help.aliyun.com/zh/model-studio/)
- [Rerank API](https://help.aliyun.com/zh/model-studio/rerank)

---

## 许可证

MIT License
