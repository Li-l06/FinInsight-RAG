from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from app.config import config


class DocumentSplitter:
    """
    段落边界感知 + 表格保护的文档切分器

    迭代过程：
    V1: RecursiveCharacterTextSplitter(fixed chunk_size=800)
        → 问题：在段落中间切断，语义不完整
    V2: MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
        → 问题：PDF 没有 Markdown 标题层级
    V3: 段落边界（\n\n）滑动窗口 + 表格块保护
    """

    TABLE_MARKER = "[表格 -"

    def __init__(self):
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # 基础切分器：当段落本身就是大块时用
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def split(self, content: str, file_path: str) -> List[Document]:
        """主切分方法"""
        if not content.strip():
            return []

        # Step 1: 按 \n\n 切分段落
        paragraphs = content.split("\n\n")

        # Step 2: 合并表格块的段落（表格被 \n\n 拆散了）
        paragraphs = self._rejoin_table_blocks(paragraphs)

        # Step 3: 段落边界滑动窗口合并
        chunks = self._merge_by_boundary(paragraphs)

        # Step 4: 转为 LangChain Document
        file_name = Path(file_path).name
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": str(Path(file_path).resolve().as_posix()),
                    "file_name": file_name,
                    "chunk_index": i,
                },
            )
            documents.append(doc)

        logger.info(f"文档切分完成: {file_name} → {len(documents)} 块")
        return documents

    def _rejoin_table_blocks(self, paragraphs: List[str]) -> List[str]:
        """将表格的多个段落块重新合并为完整表格"""
        merged = []
        in_table = False
        table_buffer = []

        for para in paragraphs:
            stripped = para.strip()
            if stripped.startswith(self.TABLE_MARKER):
                if not in_table:
                    in_table = True
                    table_buffer = [stripped]
                else:
                    table_buffer.append(stripped)
            elif in_table and stripped.startswith("|"):
                # 表格的续行（Markdown 表格行）
                table_buffer.append(stripped)
            else:
                if in_table:
                    merged.append("\n".join(table_buffer))
                    table_buffer = []
                    in_table = False
                merged.append(stripped)

        if table_buffer:
            merged.append("\n".join(table_buffer))

        return merged

    def _merge_by_boundary(self, paragraphs: List[str]) -> List[str]:
        """
        段落边界滑动窗口合并

        规则：
        - 表格块单独成 chunk，不与文本合并
        - 以 \n\n 为自然断点，合并直到接近 chunk_size
        - 重叠：下一个 chunk 包含上一个 chunk 的最后一段（而非固定字数）
        """
        chunks = []
        current = []
        current_size = 0

        for para in paragraphs:
            stripped = para.strip()
            if not stripped:
                continue

            para_size = len(stripped)

            # 表格块独立
            if stripped.startswith(self.TABLE_MARKER):
                if current:
                    chunks.append("\n\n".join(current))
                chunks.append(stripped)
                current = []
                current_size = 0
                continue

            # 如果加上当前段落会超出，先保存当前 chunk
            if current_size + para_size > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                # 重叠：保留最后一段
                current = [current[-1]] if current else []
                current_size = len(current[-1]) if current else 0

            current.append(stripped)
            current_size += para_size

        # 收尾
        if current:
            chunks.append("\n\n".join(current))

        return chunks