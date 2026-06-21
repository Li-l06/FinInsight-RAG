from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.document_parser import DocumentParser
from app.services.document_splitter import DocumentSplitter
from app.services.vector_store_service import vector_store_service
from app.services.search_service import search_service
from loguru import logger

router = APIRouter()

UPLOAD_DIR = Path("./uploads")
ALLOWED_EXTS = {".txt", ".md", ".pdf"}
MAX_SIZE = 10 * 1024 * 1024  # 10MB

splitter = DocumentSplitter()


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档 → 解析 → 切分 → 入库"""
    if not file.filename:
        raise HTTPException(400, "文件名为空")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"不支持的类型，仅支持: {', '.join(ALLOWED_EXTS)}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 标准化文件名
    safe_name = file.filename.replace(" ", "_")
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        safe_name = safe_name.replace(ch, "_")

    file_path = UPLOAD_DIR / safe_name
    content = await file.read()

    if len(content) > MAX_SIZE:
        raise HTTPException(400, f"文件过大，最大 {MAX_SIZE // 1024 // 1024}MB")

    file_path.write_bytes(content)
    logger.info(f"文件已保存: {file_path}")

    # 解析 → 切分 → 入库
    try:
        text = DocumentParser.parse(str(file_path))
        docs = splitter.split(text, str(file_path))
        vector_store_service.add_documents(docs)
        search_service.build_bm25_index()  # 重建 BM25
    except Exception as e:
        logger.error(f"文档索引失败: {e}")
        raise HTTPException(500, f"文档索引失败: {e}")

    return {
        "code": 200,
        "message": "success",
        "data": {
            "filename": safe_name,
            "chunks": len(docs),
            "size": len(content),
        },
    }


@router.get("/documents")
async def list_documents():
    """文档列表"""
    return {
        "code": 200,
        "message": "success",
        "data": vector_store_service.get_all_documents(),
    }


@router.delete("/documents/{filename:path}")
async def delete_document(filename: str):
    """删除文档及其向量"""
    file_path = UPLOAD_DIR / filename
    source = str(file_path.resolve().as_posix())

    vector_store_service.delete_by_source(source)
    if file_path.exists():
        file_path.unlink()

    search_service.build_bm25_index()
    return {"code": 200, "message": f"已删除: {filename}", "data": None}