# -*- coding: utf-8 -*-
"""使用 qwen3.5-ocr 解析图片和 PDF 页面。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.documents import Document
from openai import OpenAI

from config import (
    OCR_API_BASE,
    OCR_API_KEY,
    OCR_MAX_PIXELS,
    OCR_MAX_RETRIES,
    OCR_MAX_TOKENS,
    OCR_MIN_PIXELS,
    OCR_MODEL_NAME,
    OCR_PDF_DPI,
    OCR_REQUEST_TIMEOUT,
)

OCR_PROMPT = (
    "请完整提取图片中的全部文字，并尽量保持原有的标题、段落、列表和表格结构。"
    "表格请使用 Markdown 表格表示；不要添加解释、总结或图片中不存在的内容。"
)
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
}

_ocr_client: OpenAI | None = None


def _get_ocr_client() -> OpenAI:
    global _ocr_client
    if _ocr_client is None:
        if not OCR_API_KEY:
            raise ValueError("未配置 OCR_API_KEY、DASHSCOPE_API_KEY 或 OPENAI_API_KEY")
        if not OCR_API_BASE:
            raise ValueError("未配置 OCR_API_BASE 或 OPENAI_API_BASE")
        _ocr_client = OpenAI(
            api_key=OCR_API_KEY,
            base_url=OCR_API_BASE,
            timeout=OCR_REQUEST_TIMEOUT,
            max_retries=OCR_MAX_RETRIES,
        )
    return _ocr_client


def _ocr_image_bytes(image_bytes: bytes, mime_type: str) -> str:
    """将图片字节编码为 Data URL，并调用 qwen3.5-ocr。"""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    response = _get_ocr_client().chat.completions.create(
        model=OCR_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "min_pixels": OCR_MIN_PIXELS,
                        "max_pixels": OCR_MAX_PIXELS,
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ],
        max_tokens=OCR_MAX_TOKENS,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()


def ocr_image_file(filepath: str) -> list[Document]:
    """OCR 单张图片，返回可继续分块的 LangChain Document。"""
    path = Path(filepath)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    text = _ocr_image_bytes(path.read_bytes(), mime_type)
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={
                "source": str(path),
                "data_type": "narrative",
                "input_type": "image",
                "ocr_model": OCR_MODEL_NAME,
            },
        )
    ]


def ocr_pdf_file(filepath: str) -> list[Document]:
    """将 PDF 逐页 OCR；单页失败时回退原生文本或仅跳过该页。"""
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError("PDF OCR 需要安装 PyMuPDF：pip install PyMuPDF") from exc

    path = Path(filepath)
    documents: list[Document] = []
    zoom = OCR_PDF_DPI / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            extraction_method = "qwen3.5-ocr"
            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pixmap.tobytes("jpeg", jpg_quality=85)
                text = _ocr_image_bytes(image_bytes, "image/jpeg")
            except Exception as exc:
                # 内容审核、单页图像异常或临时接口错误不应使整份 PDF 作废。
                # 对可搜索 PDF 使用本地原生文本兜底；扫描页没有文本时只跳过该页。
                text = page.get_text("text").strip()
                if text:
                    extraction_method = "native_text_fallback"
                    print(
                        f"--- PDF 第 {page_index + 1} 页 OCR 失败，"
                        f"已回退原生文本: {path.name} ({exc}) ---"
                    )
                else:
                    print(
                        f"--- PDF 第 {page_index + 1} 页 OCR 失败且无原生文本，"
                        f"已跳过: {path.name} ({exc}) ---"
                    )
                    continue
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_index,
                        "page_number": page_index + 1,
                        "data_type": "narrative",
                        "input_type": "pdf_ocr",
                        "ocr_model": OCR_MODEL_NAME,
                        "extraction_method": extraction_method,
                    },
                )
            )
    return documents
