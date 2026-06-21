from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题")

    class Config:
        json_schema_extra = {
            "example": {"id": "session-001", "question": "Q3 毛利率提升的消费类公司有哪些？"}
        }


class ClearRequest(BaseModel):
    session_id: str = Field(..., description="会话ID", alias="sessionId")