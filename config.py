# -*- coding: utf-8 -*-
"""
@desc: 配置模块，用于加载环境变量和管理配置。
"""
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)

# --- Rerank ---
# 默认复用大模型的 OpenAI 兼容端点和密钥，也可通过独立环境变量覆盖。
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "qwen3-rerank")
RERANK_API_KEY = (
    os.getenv("RERANK_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or OPENAI_API_KEY
)
RERANK_API_BASE = os.getenv("RERANK_API_BASE")
RERANK_REQUEST_TIMEOUT = float(os.getenv("RERANK_REQUEST_TIMEOUT", "60"))
HYBRID_DENSE_CANDIDATES = int(os.getenv("HYBRID_DENSE_CANDIDATES", "30"))
HYBRID_SPARSE_CANDIDATES = int(os.getenv("HYBRID_SPARSE_CANDIDATES", "30"))
HYBRID_RERANK_CANDIDATES = int(os.getenv("HYBRID_RERANK_CANDIDATES", "30"))

# --- 文档分块 ---
# 这里按字符数而非模型 Token 数计量。
DOCUMENT_CHUNK_SIZE = int(os.getenv("DOCUMENT_CHUNK_SIZE", "1000"))
DOCUMENT_CHUNK_OVERLAP = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "200"))
RETRIEVAL_CHUNK_SIZE = int(os.getenv("RETRIEVAL_CHUNK_SIZE", "350"))
RETRIEVAL_CHUNK_OVERLAP = int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "50"))

# --- 上下文工程 ---
CONTEXT_COMPRESS_THRESHOLD = int(os.getenv("CONTEXT_COMPRESS_THRESHOLD", "12000"))
CONTEXT_TARGET_CHARS = int(os.getenv("CONTEXT_TARGET_CHARS", "8000"))
CONVERSATION_SUMMARY_START_TURNS = int(os.getenv("CONVERSATION_SUMMARY_START_TURNS", "10"))
CONVERSATION_SUMMARY_INTERVAL = int(os.getenv("CONVERSATION_SUMMARY_INTERVAL", "5"))
CONVERSATION_RECENT_TURNS = int(os.getenv("CONVERSATION_RECENT_TURNS", "5"))

# --- OCR ---
OCR_MODEL_NAME = os.getenv("OCR_MODEL_NAME", "qwen3.5-ocr")
OCR_API_KEY = os.getenv("OCR_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or OPENAI_API_KEY
OCR_API_BASE = os.getenv("OCR_API_BASE") or OPENAI_API_BASE
OCR_REQUEST_TIMEOUT = float(os.getenv("OCR_REQUEST_TIMEOUT", "180"))
OCR_MAX_RETRIES = int(os.getenv("OCR_MAX_RETRIES", "3"))
OCR_MAX_TOKENS = int(os.getenv("OCR_MAX_TOKENS", "16384"))
OCR_PDF_DPI = int(os.getenv("OCR_PDF_DPI", "160"))
OCR_MIN_PIXELS = int(os.getenv("OCR_MIN_PIXELS", str(32 * 32 * 3)))
OCR_MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", str(32 * 32 * 8192)))

# 如果使用Qwen，请取消下面的注释
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# --- LLM --- 
# 使用的模型名称 (例如 "gpt-4o", "deepseek-v3-1-terminus")
LLM_MODEL_NAME = "qwen3.6-plus"
# LLM_MODEL_NAME = "qwen-turbo"

# 单次模型请求超时（秒）。长上下文行程生成通常会超过 60 秒。
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))
# 网络超时、连接失败等瞬时错误的自动重试次数。
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# 如果您使用自定义的、兼容OpenAI API的端点（例如Ollama, LocalAI等），请在此处设置其URL
# 例如: "http://localhost:11434/v1"
# --- Embedding ---

# 选择嵌入模型的提供商: 'openai' 或 'local'
# 'openai': 使用兼容OpenAI API的嵌入模型 (包括OpenAI官方、Azure、Ollama等)。
# 'local': 使用本地句向量模型 (SentenceTransformers/HuggingFace)。
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# -- OpenAI 嵌入模型配置 (当 EMBEDDING_PROVIDER = 'openai') --
# 如果嵌入模型的API地址与主模型不同，请在此处设置
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", None)
EMBEDDING_API_KEY = (
    os.getenv("EMBEDDING_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or OPENAI_API_KEY
)
# 使用的嵌入模型名称。
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")

# -- 本地嵌入模型配置 (当 EMBEDDING_PROVIDER = 'local') --
# 指定本地模型的路径或HuggingFace模型库的ID
# 例如: 'sentence-transformers/all-MiniLM-L6-v2'
LOCAL_EMBEDDING_MODEL_PATH = os.getenv("LOCAL_EMBEDDING_MODEL_PATH", "BAAI/bge-m3")

# --- Excel 数据加载配置 ---
# 在加载Excel文件时，指定哪些列应该被提取为文档的元数据。
# 这些列的值将作为键值对存储在向量库中，用于后续的过滤或更精确的检索。
EXCEL_METADATA_COLUMNS = [
    "doc_id",
    "doc_type",
    "region",
    "destination",
    "full_destination",
    "attraction",
    "source_row",
]
