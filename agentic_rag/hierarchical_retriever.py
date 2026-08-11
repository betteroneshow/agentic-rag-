# -*- coding: utf-8 -*-
"""本地知识库的分层检索与混合检索。

候选召回由 BM25 稀疏检索和 ChromaDB 稠密向量检索组成，使用 RRF
融合两路排名，最后通过 qwen3-rerank 进行精排。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from threading import Lock

import chromadb
import httpx
from langchain_core.documents import Document

from agentic_rag.chains import get_embedding_function
from config import (
    HYBRID_DENSE_CANDIDATES,
    HYBRID_RERANK_CANDIDATES,
    HYBRID_SPARSE_CANDIDATES,
    OPENAI_API_BASE,
    RERANK_API_BASE,
    RERANK_API_KEY,
    RERANK_MODEL_NAME,
    RERANK_REQUEST_TIMEOUT,
)

PERSIST_PATH = "chroma_db"
SUMMARY_COLLECTION_NAME = "doc_summaries"
CHUNK_COLLECTION_NAME = "doc_chunks"
RETRIEVAL_COLLECTION_NAME = "retrieval_chunks"
RRF_K = 60
RERANK_MAX_DOCUMENT_CHARS = 12_000

client = chromadb.PersistentClient(path=PERSIST_PATH)
embedding_function = get_embedding_function()
summary_collection = client.get_collection(
    SUMMARY_COLLECTION_NAME, embedding_function=embedding_function
)
chunk_collection = client.get_collection(
    CHUNK_COLLECTION_NAME, embedding_function=embedding_function
)
retrieval_collection = client.get_or_create_collection(
    RETRIEVAL_COLLECTION_NAME, embedding_function=embedding_function
)

def _search_collection():
    """新索引检索子块；旧索引未重建时兼容回退到父块。"""
    return retrieval_collection if retrieval_collection.count() else chunk_collection


def _tokenize(text: str) -> list[str]:
    """为中英文混合旅游文本生成 BM25 词元，无需额外分词依赖。"""
    text = str(text or "").lower()
    tokens: list[str] = []
    for segment in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[a-z0-9]+", segment):
            tokens.append(segment)
            continue
        tokens.extend(segment)
        tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return tokens


class BM25Index:
    """轻量 BM25Okapi 索引，适合当前 Chroma 文档集合规模。"""

    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        # 标题性字段在稀疏检索中应比普通正文拥有更高权重。
        field_weights = {
            "full_destination": 4,
            "destination": 4,
            "attraction": 3,
            "region": 2,
            "doc_type": 2,
        }
        self.term_frequencies = []
        for document, metadata in zip(documents, metadatas):
            frequencies = Counter(_tokenize(document))
            for field, weight in field_weights.items():
                value = metadata.get(field)
                if value:
                    frequencies.update(_tokenize(str(value)) * weight)
            self.term_frequencies.append(frequencies)
        self.lengths = [sum(tf.values()) for tf in self.term_frequencies]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequencies: Counter[str] = Counter()
        for tf in self.term_frequencies:
            document_frequencies.update(tf.keys())
        corpus_size = len(documents)
        self.idf = {
            term: math.log(1.0 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(
        self,
        query: str,
        limit: int,
        allowed_sources: set[str] | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[dict]:
        query_terms = _tokenize(query)
        if not query_terms or not self.documents:
            return []

        k1, b = 1.5, 0.75
        scored: list[tuple[float, int]] = []
        for index, (tf, length, metadata) in enumerate(
            zip(self.term_frequencies, self.lengths, self.metadatas)
        ):
            if allowed_sources is not None and metadata.get("source") not in allowed_sources:
                continue
            if metadata_filter and any(
                metadata.get(field) != value for field, value in metadata_filter.items()
            ):
                continue
            score = 0.0
            for term in query_terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                normalizer = frequency + k1 * (
                    1.0 - b + b * length / max(self.avg_length, 1.0)
                )
                score += self.idf.get(term, 0.0) * frequency * (k1 + 1.0) / normalizer
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "id": self.ids[index],
                "document": self.documents[index],
                "metadata": self.metadatas[index],
                "bm25_score": score,
            }
            for score, index in scored[:limit]
        ]


_bm25_index: BM25Index | None = None
_bm25_lock = Lock()


def _get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        with _bm25_lock:
            if _bm25_index is None:
                records = _search_collection().get(include=["documents", "metadatas"])
                ids = [str(item) for item in records.get("ids", [])]
                documents = [str(item or "") for item in records.get("documents", [])]
                metadatas = [item or {} for item in records.get("metadatas", [])]
                _bm25_index = BM25Index(ids, documents, metadatas)
                print(f"--- BM25 索引已加载，共 {len(ids)} 个文本块 ---")
    return _bm25_index


def refresh_bm25_index() -> None:
    """知识库重新入库后调用，使下一次检索重建 BM25 索引。"""
    global _bm25_index
    with _bm25_lock:
        _bm25_index = None


def _build_chroma_filter(
    sources: list[str] | None,
    metadata_filter: dict[str, str] | None,
) -> dict | None:
    clauses: list[dict] = []
    if sources:
        clauses.append({"source": {"$in": sources}})
    for field, value in (metadata_filter or {}).items():
        clauses.append({field: {"$eq": value}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _infer_metadata_filter(query: str, index: BM25Index) -> dict[str, str]:
    """从查询中匹配知识库已有目的地，生成安全的精确过滤条件。"""
    matches: list[tuple[int, str, str]] = []
    for metadata in index.metadatas:
        for field in ("destination", "full_destination"):
            value = str(metadata.get(field) or "").strip()
            if not value:
                continue
            aliases = {value, value.rstrip("省市区县")}
            if any(alias and alias in query for alias in aliases):
                matches.append((max(len(alias) for alias in aliases), field, value))
    if not matches:
        return {}
    _, field, value = max(matches, key=lambda item: item[0])
    return {field: value}


def _dense_search(
    query: str,
    limit: int,
    sources: list[str] | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> list[dict]:
    active_collection = _search_collection()
    collection_size = active_collection.count()
    if collection_size == 0:
        return []
    query_args = {
        "query_texts": [query],
        "n_results": min(limit, collection_size),
        "include": ["documents", "metadatas", "distances"],
    }
    where_filter = _build_chroma_filter(sources, metadata_filter)
    if where_filter:
        query_args["where"] = where_filter
    results = active_collection.query(**query_args)
    return [
        {
            "id": str(item_id),
            "document": document,
            "metadata": metadata or {},
            "dense_distance": float(distance),
        }
        for item_id, document, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        )
    ]


def _rrf_fuse(dense: list[dict], sparse: list[dict], limit: int) -> list[dict]:
    """使用 Reciprocal Rank Fusion 融合不同量纲的召回结果。"""
    candidates: dict[str, dict] = {}
    for source_name, results in (("dense", dense), ("bm25", sparse)):
        for rank, item in enumerate(results, start=1):
            item_id = item["id"]
            candidate = candidates.setdefault(
                item_id,
                {
                    "id": item_id,
                    "document": item["document"],
                    "metadata": dict(item.get("metadata") or {}),
                    "fusion_score": 0.0,
                },
            )
            candidate["fusion_score"] += 1.0 / (RRF_K + rank)
            candidate[f"{source_name}_rank"] = rank
            if source_name == "dense":
                candidate["dense_distance"] = item.get("dense_distance")
            else:
                candidate["bm25_score"] = item.get("bm25_score")

    return sorted(
        candidates.values(), key=lambda item: item["fusion_score"], reverse=True
    )[:limit]


def _rerank_endpoint() -> str:
    base_url = RERANK_API_BASE or OPENAI_API_BASE
    if not base_url:
        base_url = "https://dashscope.aliyuncs.com/compatible-api/v1"

    # 百炼工作空间的聊天/Embedding 与 rerank 使用不同的兼容路径：
    #   compatible-mode/v1 -> chat、embeddings
    #   compatible-api/v1  -> qwen3-rerank
    # 当复用 OPENAI_API_BASE 时自动转换，显式 RERANK_API_BASE 仍拥有最高优先级。
    if not RERANK_API_BASE:
        base_url = base_url.replace("/compatible-mode/v1", "/compatible-api/v1")

    base_url = base_url.rstrip("/")
    if base_url.endswith("/reranks"):
        return base_url
    return f"{base_url}/reranks"


def _qwen_rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    if not candidates:
        return []
    if not RERANK_API_KEY:
        print("--- 未配置 RERANK_API_KEY，使用 RRF 融合排序 ---")
        return candidates[:top_n]

    documents = []
    for item in candidates:
        metadata = item.get("metadata") or {}
        prefix = "\n".join(
            f"{label}：{metadata[field]}"
            for field, label in (
                ("destination", "目的地"),
                ("doc_type", "资料类型"),
                ("attraction", "景点"),
                ("record_level", "记录层级"),
            )
            if metadata.get(field)
        )
        documents.append(f"{prefix}\n{item['document']}"[:RERANK_MAX_DOCUMENT_CHARS])
    try:
        response = httpx.post(
            _rerank_endpoint(),
            headers={
                "Authorization": f"Bearer {RERANK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL_NAME,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "instruct": "Given a travel planning query, retrieve passages that directly answer the query.",
            },
            timeout=RERANK_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or payload.get("output", {}).get("results") or []
        reranked: list[dict] = []
        for result in results:
            index = int(result["index"])
            if not 0 <= index < len(candidates):
                continue
            candidate = dict(candidates[index])
            candidate["rerank_score"] = float(result["relevance_score"])
            reranked.append(candidate)
        if reranked:
            return reranked
        raise ValueError("rerank 响应中没有有效 results")
    except Exception as exc:
        print(f"--- qwen3-rerank 调用失败，回退到 RRF 排序: {exc} ---")
        return candidates[:top_n]


def _diversify_results(candidates: list[dict], limit: int) -> list[dict]:
    """去除重复实体，并优先覆盖不同资料类型。"""
    deduplicated: list[dict] = []
    seen_entities: set[tuple] = set()
    for item in candidates:
        metadata = item.get("metadata") or {}
        entity_key = (
            metadata.get("destination"),
            metadata.get("attraction"),
            metadata.get("doc_type"),
            metadata.get("record_level"),
        )
        if metadata.get("attraction") and entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        deduplicated.append(item)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    seen_types: set[str] = set()
    # 第一轮优先覆盖景点、交通、美食、住宿等不同主题。
    for item in deduplicated:
        doc_type = str((item.get("metadata") or {}).get("doc_type") or "综合")
        if doc_type in seen_types:
            continue
        selected.append(item)
        selected_ids.add(item["id"])
        seen_types.add(doc_type)
        if len(selected) >= limit:
            return selected
    # 第二轮严格遵循 rerank 排名补足数量。
    for item in deduplicated:
        if item["id"] in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _hybrid_chunk_search(
    query: str,
    n_results: int,
    sources: list[str] | None = None,
) -> list[Document]:
    bm25_index = _get_bm25_index()
    metadata_filter = _infer_metadata_filter(query, bm25_index)
    if metadata_filter:
        print(f"--- 应用表格元数据过滤: {metadata_filter} ---")
    dense = _dense_search(
        query, HYBRID_DENSE_CANDIDATES, sources, metadata_filter
    )
    sparse = bm25_index.search(
        query,
        HYBRID_SPARSE_CANDIDATES,
        set(sources) if sources else None,
        metadata_filter,
    )
    fused = _rrf_fuse(dense, sparse, HYBRID_RERANK_CANDIDATES)
    rerank_limit = min(len(fused), max(n_results * 3, n_results))
    reranked = _qwen_rerank(query, fused, rerank_limit)
    reranked = _diversify_results(reranked, n_results)

    # 小块完成召回和 rerank 后，按 parent_id 一次性取回连贯父块。
    parent_ids = list(dict.fromkeys(
        str(item.get("metadata", {}).get("parent_id"))
        for item in reranked if item.get("metadata", {}).get("parent_id")
    ))
    parent_map: dict[str, tuple[str, dict]] = {}
    if parent_ids:
        parent_result = chunk_collection.get(
            ids=parent_ids, include=["documents", "metadatas"]
        )
        parent_map = {
            str(item_id): (document, metadata or {})
            for item_id, document, metadata in zip(
                parent_result.get("ids", []), parent_result.get("documents", []),
                parent_result.get("metadatas", []),
            )
        }

    documents: list[Document] = []
    returned_parents: set[str] = set()
    for item in reranked:
        metadata = dict(item["metadata"])
        parent_id = str(metadata.get("parent_id") or "")
        if parent_id and parent_id in returned_parents:
            continue
        for key in (
            "dense_rank",
            "bm25_rank",
            "dense_distance",
            "bm25_score",
            "fusion_score",
            "rerank_score",
        ):
            if key in item and item[key] is not None:
                metadata[key] = item[key]
        content = item["document"]
        if parent_id and parent_id in parent_map:
            content, parent_metadata = parent_map[parent_id]
            metadata = {**parent_metadata, **metadata, "matched_child_id": item["id"]}
            returned_parents.add(parent_id)
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def hierarchical_retriever(query: str, n_docs: int = 3, n_chunks: int = 5) -> list[Document]:
    """先用摘要层限定文档范围，再对块执行混合召回与精排。"""
    print("--- 执行分层混合检索 ---")
    summary_count = summary_collection.count()
    if summary_count == 0:
        return []
    summary_results = summary_collection.query(
        query_texts=[query],
        n_results=min(n_docs, summary_count),
        include=["metadatas"],
    )
    sources = [
        metadata.get("source")
        for metadata in summary_results.get("metadatas", [[]])[0]
        if metadata and metadata.get("source")
    ]
    if not sources:
        return []
    print(f"--- 摘要层命中文档源: {sources} ---")
    return _hybrid_chunk_search(query, n_chunks, sources)


def direct_chunk_retriever(query: str, n_chunks: int = 5) -> list[Document]:
    """在全部文本块上执行 BM25 + 稠密向量 + qwen3-rerank。"""
    print("--- 执行混合检索: BM25 + text-embedding-v4 + qwen3-rerank ---")
    return _hybrid_chunk_search(query, n_chunks)
