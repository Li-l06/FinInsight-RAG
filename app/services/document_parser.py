from pathlib import Path
from typing import List, Tuple, Optional
import pdfplumber
from loguru import logger


class DocumentParser:
    """统一文档解析器"""

    @staticmethod
    def parse(file_path: str) -> str:
        """根据扩展名分派解析"""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        elif ext in (".txt", ".md"):
            return Path(file_path).read_text(encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """解析 PDF：提取文本 + 表格转 Markdown"""
        all_parts: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 1. 提取文本
                text = page.extract_text()
                if text and text.strip():
                    all_parts.append(text.strip())

                # 2. 提取表格 → Markdown
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 0:
                        md_table = DocumentParser._table_to_markdown(table)
                        all_parts.append(f"[表格 - 第{page_num}页]\n{md_table}")

        content = "\n\n".join(all_parts)
        logger.info(f"PDF 解析完成: {file_path}, {len(pdf.pages)} 页, {len(content)} 字")
        return content

    @staticmethod
    def _table_to_markdown(table: List[List[Optional[str]]]) -> str:
        """将表格数据转为 Markdown 表格"""
        if not table:
            return ""

        # 过滤全空行
        rows = [row for row in table if any(cell and str(cell).strip() for cell in row)]
        if not rows:
            return ""

        # 清理单元格
        clean = [[str(cell).strip() if cell else "" for cell in row] for row in rows]

        # 计算列宽
        col_count = max(len(row) for row in clean)
        for row in clean:
            while len(row) < col_count:
                row.append("")

        lines = []
        # 表头
        lines.append("| " + " | ".join(clean[0]) + " |")
        # 分隔行
        lines.append("| " + " | ".join(["---"] * col_count) + " |")
        # 数据行
        for row in clean[1:]:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)