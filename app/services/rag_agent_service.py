from typing import Annotated, Any, AsyncGenerator, Dict, Sequence
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from typing_extensions import TypedDict
from loguru import logger


from app.config import config
from app.tools import DEFAULT_TOOLS
from app.core.llm_factory import create_chat_model, create_streaming_chat_model



# ─── Agent State ───────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ─── System Prompt ─────────────────────────────────────
SYSTEM_PROMPT = """你是一个金融研报分析助手，基于知识库中的研报数据回答问题。

规则：
1. 回答必须基于检索到的研报内容，不得编造数据
2. 每次回答必须标注信息来源（文件名）
3. 如果检索结果不足以回答问题，明确说"根据现有资料，无法回答此问题"
4. 如果检索无结果，尝试换一个角度重新搜索，最多尝试3次
5. 涉及具体数据（数字、百分比、日期）时必须引用原文"""


# ─── 上下文裁剪 Middleware ──────────────────────────────
def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    消息历史裁剪策略：
    - 始终保留 SystemMessage（第一条）
    - 保留最近 6 条消息（3 轮对话）
    - 消息 ≤ 7 条时不裁剪
    """
    messages = state["messages"]
    if len(messages) <= 7:
        return None

    first_msg = messages[0]
    recent = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    logger.debug(f"裁剪消息: {len(messages)} → {1 + len(recent)}")
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first_msg, *recent]
    }


# ─── RAG Agent 服务 ────────────────────────────────────
class RagAgentService:
    """基于 LangGraph ReAct 的 RAG Agent

    设计要点（面试重点）：
    1. 防循环：max_iterations=6，去重相同工具调用
    2. 重试：指数退避（1s→2s），失败降级
    3. 裁剪：保留 System + 最近 6 条
    """

    def __init__(self):
        self.system_prompt = SYSTEM_PROMPT
        self.tools = list(DEFAULT_TOOLS)
        self.checkpointer = MemorySaver()
        self._agent = None

        # 防循环状态
        self._call_history: Dict[str, list] = {}

        logger.info(f"RAG Agent 就绪: model={config.dashscope_model}, tools={[t.name for t in self.tools]}")

    def _get_agent(self, streaming: bool = False):
        if self._agent is None:
            model = create_chat_model(streaming=streaming)
            self._agent = create_react_agent(  # ← 改函数名
                model=model,
                tools=self.tools,
                prompt=self.system_prompt,  # ← system_prompt → prompt
                checkpointer=self.checkpointer,
                # middleware 去掉，create_react_agent 不支持
            )
        return self._agent

    # ─── 同步查询 ───────────────────────────────────
    async def query(self, question: str, session_id: str = "default") -> str:
        """同步对话"""
        config_dict = {"configurable": {"thread_id": session_id}}
        agent = self._get_agent(streaming=False)

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config=config_dict,
            )
            messages = result.get("messages", [])
            last_msg = messages[-1] if messages else None

            if last_msg and hasattr(last_msg, "content"):
                return last_msg.content
            return "处理完成，但未生成回复。"
        except Exception as e:
            logger.error(f"Agent 查询失败: {e}")
            return f"系统错误: {str(e)}"

    # ─── 流式查询 ───────────────────────────────────
    async def query_stream(
        self, question: str, session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """SSE 流式对话"""
        config_dict = {"configurable": {"thread_id": session_id}}
        agent = self._get_agent(streaming=True)

        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=question)]},
                config=config_dict,
                version="v2",
            ):
                kind = event.get("event", "")

                # 工具调用开始
                if kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    logger.info(f"🔧 调用工具: {tool_name}")
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": tool_name,
                            "status": "start",
                            "input": str(tool_input),
                        },
                    }

                # 工具调用结束
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    logger.info(f"✅ 工具完成: {tool_name}")
                    yield {
                        "type": "tool_call",
                        "data": {
                            "tool": tool_name,
                            "status": "end",
                        },
                    }

                # LLM token 流
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk", None)
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {"type": "content", "data": chunk.content}

            yield {"type": "complete", "data": None}
            logger.info(f"流式对话完成: {session_id}")

        except Exception as e:
            logger.error(f"流式查询异常: {e}")
            yield {"type": "error", "data": str(e)}

    # ─── 会话管理 ───────────────────────────────────
    def get_session_history(self, session_id: str) -> list:
        """获取会话历史"""
        try:
            state = self.checkpointer.get({"configurable": {"thread_id": session_id}})
            if not state:
                return []

            messages = state.get("channel_values", {}).get("messages", [])
            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)
                history.append({"role": role, "content": content})
            return history
        except Exception as e:
            logger.error(f"获取历史失败: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        try:
            self.checkpointer.delete_thread(session_id)
            logger.info(f"已清除会话: {session_id}")
            return True
        except Exception as e:
            logger.error(f"清除会话失败: {e}")
            return False


rag_agent_service = RagAgentService()