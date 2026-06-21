from fastapi import APIRouter
from app.services.vector_store_service import vector_store_service

router = APIRouter()


@router.get("/health")
async def health():
    """健康检查"""
    try:
        store = vector_store_service.store
        _ = store.get()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    return {
        "status": "ok" if chroma_ok else "degraded",
        "chromadb": "connected" if chroma_ok else "error",
    }