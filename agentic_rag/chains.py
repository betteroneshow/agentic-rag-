# -*- coding: utf-8 -*-
"""
@desc: LLM链模块

定义了系统中使用的各种LLM链，例如查询路由、查询重写和答案评估。
"""

try:
    import torch
except ImportError:  # pragma: no cover - optional for OpenAI embedding mode
    torch = None
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - optional for OpenAI embedding mode
    HuggingFaceEmbeddings = None
from chromadb.utils import embedding_functions
from chromadb.api.types import Documents, Embeddings

from config import (
    LLM_MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY, LLM_REQUEST_TIMEOUT, LLM_MAX_RETRIES,
    EMBEDDING_PROVIDER, EMBEDDING_API_BASE, EMBEDDING_API_KEY,
    EMBEDDING_MODEL_NAME, LOCAL_EMBEDDING_MODEL_PATH
)

# --- LLM 初始化 ---
# 构造LLM参数
llm_params = {
    "model": LLM_MODEL_NAME,
    "temperature": 0,
    "api_key": OPENAI_API_KEY or "dummy-key",
    "request_timeout": LLM_REQUEST_TIMEOUT,
    "max_retries": LLM_MAX_RETRIES,
}
# 如果配置了自定义API地址，则使用它
if OPENAI_API_BASE:
    llm_params["base_url"] = OPENAI_API_BASE

# 使用 config.py 中定义的模型和可选的自定义API地址
llm = ChatOpenAI(**llm_params)

class LangChainEmbeddingAdapter:
    """Adapt LangChain embeddings to Chroma's embedding function interface."""

    def __init__(self, embeddings, model_name: str, batch_size: int = 10):
        self.embeddings = embeddings
        self.model_name = model_name
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self.embeddings.embed_documents(texts[i:i + self.batch_size]))
        return vectors

    def embed_query(self, input):
        if isinstance(input, str):
            return self.embeddings.embed_documents([input])[0]
        return self.__call__(input)

    def name(self) -> str:
        return f"langchain-openai-{self.model_name}"

def get_embedding_function():
    """根据配置获取嵌入模型函数。"""
    if EMBEDDING_PROVIDER == 'openai':
        print("--- 使用OpenAI嵌入模型 ---")
        embedding_params = {
            "model": EMBEDDING_MODEL_NAME,
            "api_key": EMBEDDING_API_KEY or "dummy-key",
            # Some OpenAI-compatible providers reject token-id inputs and only
            # accept raw text strings in the embeddings request.
            "check_embedding_ctx_length": False,
        }
        # 优先使用独立的嵌入模型API地址，否则回退到主API地址
        api_base = EMBEDDING_API_BASE or OPENAI_API_BASE
        if api_base:
            embedding_params["base_url"] = api_base
        return LangChainEmbeddingAdapter(
            OpenAIEmbeddings(**embedding_params),
            EMBEDDING_MODEL_NAME,
        )
    
    elif EMBEDDING_PROVIDER == 'local':
        if torch is None:
            raise ImportError("Local embedding mode requires torch to be installed.")
        if HuggingFaceEmbeddings is None:
            raise ImportError("Local embedding mode requires langchain-huggingface to be installed.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- 使用ChromaDB原生本地嵌入模型: {LOCAL_EMBEDDING_MODEL_PATH} (设备: {device}) ---")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=LOCAL_EMBEDDING_MODEL_PATH,
            device=device
        )
        
    else:
        raise ValueError(f"未知的嵌入模型提供商: {EMBEDDING_PROVIDER}。请选择 'openai' 或 'local'。")

# --- 输出数据结构定义 ---

class RouteQuery(BaseModel):
    """根据用户问题决定路由策略。"""
    datasource: str = Field(description="根据问题类型，从 ‘direct_chunk_search’, ‘hierarchical_search’, ‘web_search’, ‘direct’ 中选择一种最合适的路由策略。")

class RewriteQuery(BaseModel):
    """一个经过优化的、更适合检索的用户问题版本。"""
    rewritten_query: str = Field(description="对原始问题的改写，使其更适合搜索引擎或向量数据库。")

class RelevanceGrade(BaseModel):
    """评估答案是否与原始问题相关。"""
    is_relevant: bool = Field(description="布尔值，表示答案是否相关。")

class DocumentRelevanceGrade(BaseModel):
    """评估一组文档是否与用户问题相关。"""
    is_relevant: bool = Field(description="布尔值，表示这组文档是否包含足够的信息来回答问题。")

# --- LLM 链定义 ---

def get_document_relevance_grader_chain():
    """获取文档相关性评估链"""
    parser = JsonOutputParser(pydantic_object=DocumentRelevanceGrade)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位信息相关性评估专家。请根据用户问题，判断下面提供的一组文档是否包含足够的相关信息来回答该问题。必须只返回符合格式要求的 JSON，不要只返回 True 或 False。\n{format_instructions}"),
        ("human", "用户问题: {query}\n\n检索到的文档:\n{documents}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_query_router_chain():
    """获取查询路由链（已升级为智能路由）"""
    parser = JsonOutputParser(pydantic_object=RouteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位旅游规划助手的查询路由专家。请仔细分析用户的问题，并参考下面可能相关的历史记忆，然后根据指南选择最合适的检索策略。\n\n--- 历史记忆 ---\n{memories}\n--- 历史记忆结束 ---\n\n决策指南：\n1. 如果问题涉及旅游目的地、景点、交通安排、住宿推荐、美食、小贴士、旅行攻略，优先选择 ‘direct_chunk_search’，因为本地旅游知识库包含这些结构化攻略信息。\n2. 如果问题是在总结一个宽泛目的地或攻略主题，也可以选择 ‘hierarchical_search’，但具体目的地、景点、路线、交通问题优先选择 ‘direct_chunk_search’。\n3. 如果问题需要**实时或最新的信息**，例如今天/明天/下周天气、实时票价、营业状态、交通管制、最新新闻，请选择 ‘web_search’。\n4. 如果问题是**简单的对话或问候**（例如‘你好’），请选择 ‘direct’。\n\n{format_instructions}"),
        ("human", "问题: {query}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_initial_rewriter_chain():
    """获取初始查询重写链"""
    parser = JsonOutputParser(pydantic_object=RewriteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位查询优化专家。结合对话摘要和最近对话，消解‘那个、它、便宜的’等指代，将问题改写为独立、明确、适合检索的查询。不要添加上下文中不存在的事实。\n{format_instructions}"),
        ("human", "<conversation_summary>{conversation_summary}</conversation_summary>\n<recent_history>{recent_history}</recent_history>\n<original_query>{query}</original_query>")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_context_compressor_chain():
    """仅保留能够回答问题的证据，并维持来源编号。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是检索上下文压缩器。围绕用户问题删除重复和无关内容，保留事实、数字、限制条件和来源编号；不得补充原文没有的信息。输出精简的中文上下文，不要回答问题。"),
        ("human", "<query>{query}</query>\n<context>{context}</context>\n目标长度不超过 {target_chars} 个字符。")
    ])
    return prompt | llm

def get_conversation_summarizer_chain():
    """生成可替代早期对话的滚动摘要。"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你负责维护 Agent 短期记忆。合并旧摘要和早期对话，保留用户目标、偏好、已确认事实、约束、未完成事项和重要结论；省略寒暄与重复表述，不得虚构。"),
        ("human", "<old_summary>{old_summary}</old_summary>\n<history>{history}</history>")
    ])
    return prompt | llm

def get_correctional_rewriter_chain():
    """获取修正性查询重写链"""
    parser = JsonOutputParser(pydantic_object=RewriteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位查询优化专家。用户之前的查询未能得到相关的答案。请分析原始问题和这个不满意的答案，然后将问题改写得更清晰、更具体，以便更好地检索。\n{format_instructions}"),
        ("human", "原始问题: {query}\n不满意的答案: {response}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_relevance_grader_chain():

    """获取相关性评估链"""
    parser = JsonOutputParser(pydantic_object=RelevanceGrade)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位信息相关性评估专家。请根据用户问题，判断提供的答案是否相关。必须只返回符合格式要求的 JSON，不要只返回 True 或 False。\n{format_instructions}"),
        ("human", "问题: {query}\n答案: {response}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_summarizer_chain():
    """获取文档摘要链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个文档摘要专家。请为以下文档生成一个简洁但全面的摘要，摘要应捕获所有核心主题、关键实体和结论，以便后续能通过摘要判断文档与用户问题的相关性。"),
        ("human", "文档内容:\n\n{document_content}")
    ])
    return prompt | llm

class MemoryToSave(BaseModel):
    """用于存储到长期记忆库的结构化信息。"""
    should_save: bool = Field(description="本次对话是否包含值得存入长期记忆的信息。")
    text: str = Field(description="需要被记住的关键信息；无需保存时返回空字符串。")
    type: str = Field(description="记忆类型；从 ['fact', 'preference', 'conclusion'] 中选择，无需保存时使用 'fact'。")
    importance: int = Field(description="记忆的重要性评分；保存时为1到10，无需保存时为0。")

def get_memory_consolidation_chain():
    """获取记忆提炼链，用于在对话结束后总结并形成结构化记忆。"""
    parser = JsonOutputParser(pydantic_object=MemoryToSave)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个记忆提炼专家。请分析以下对话，并从中提取最值得长期记住的核心信息。"
                   "如果存在值得未来参考的信息，将 should_save 设为 true，并填写 text、type 和1到10的 importance。"
                   "如果不存在，将 should_save 设为 false、text 设为空字符串、type 设为 fact、importance 设为0。"
                   "无论是否需要保存，都必须只返回符合要求的 JSON，不能返回普通文本。\n\n{format_instructions}"),
        ("human", "对话历史:\n\n{conversation_history}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser
