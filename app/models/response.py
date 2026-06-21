from pydantic import BaseModel
from typing import Optional


class ApiResponse(BaseModel):
    status: str = "success"
    message: str = ""
    data: Optional[object] = None