import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from app.models.request import ChatRequest, ClearRequest
from app.services.rag_agent_service import rag_agent_service
from loguru import logger

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """同步对话"""
    logger.info(f"[{request.id}] 同步对话: {request.question}")
    try:
        answer = await rag_agent_service.query(request.question, session_id=request.id)
        return {
            "code": 200,
            "message": "success",
            "data": {"answer": answer},
        }
    except Exception as e:
        logger.error(f"对话失败: {e}")
        return {
            "code": 500,
            "message": str(e),
            "data": None,
        }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式对话"""
    logger.info(f"[{request.id}] 流式对话: {request.question}")

    async def event_gen():
        async for chunk in rag_agent_service.query_stream(
            request.question, session_id=request.id
        ):
            yield {
                "event": "message",
                "data": json.dumps(chunk, ensure_ascii=False),
            }

    return EventSourceResponse(event_gen())


@router.post("/chat/clear")
async def clear(request: ClearRequest):
    """清除会话"""
    success = rag_agent_service.clear_session(request.session_id)
    return {
        "code": 200 if success else 500,
        "message": "已清除" if success else "清除失败",
        "data": None,
    }


@router.get("/chat/session/{session_id}")
async def get_session(session_id: str):
    """获取会话历史"""
    history = rag_agent_service.get_session_history(session_id)
    return {
        "code": 200,
        "message": "success",
        "data": {"session_id": session_id, "history": history},
    }