# -*- coding: utf-8 -*-
"""长期记忆：Repository 关系事实源 + ChromaDB 语义索引。"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import sys
from typing import Any

import chromadb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic_rag.chains import get_embedding_function
from agentic_rag.memory_repository import get_memory_repository

PERSIST_PATH = "chroma_db"
MEMORY_COLLECTION_NAME = "long_term_memory"
DEFAULT_USER_ID = "default"
DEDUP_DISTANCE_THRESHOLD = float(os.getenv("MEMORY_DEDUP_DISTANCE_THRESHOLD", "0.12"))


def _now() -> dt.datetime:
    return dt.datetime.now()


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", str(text or "").lower())


def _repository():
    return get_memory_repository()


def _collection(create: bool = False):
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    embedding = get_embedding_function()
    if create:
        return client.get_or_create_collection(MEMORY_COLLECTION_NAME, embedding_function=embedding)
    return client.get_collection(MEMORY_COLLECTION_NAME, embedding_function=embedding)


def initialize_memory_db() -> None:
    """初始化当前配置的关系库并修复语义索引。"""
    print("--- 初始化长期记忆库 ---")
    _repository().initialize()

    collection = _collection(create=True)
    repair_memory_index(collection=collection)
    print("--- 长期记忆库初始化完成 ---")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    get = row.__getitem__
    return {
        "type": get("type"), "importance": int(get("importance")),
        "sqlite_id": int(get("id")), "user_id": get("user_id"),
        "status": get("status"), "source_type": get("source_type"),
        "confidence": float(get("confidence")),
    }


def repair_memory_index(collection=None) -> dict[str, int]:
    """补齐 SQLite/Chroma 缺失项，并清除没有 SQLite 记录的孤儿向量。"""
    collection = collection or _collection(create=True)
    repository = _repository()
    rows = repository.list_not_deleted()
    sqlite_ids = {str(row["id"]) for row in rows}
    chroma = collection.get(include=["metadatas"])
    chroma_ids = set(map(str, chroma.get("ids", [])))
    chroma_meta = {str(item_id): (meta or {}) for item_id, meta in zip(chroma.get("ids", []), chroma.get("metadatas", []))}
    stale = [
        row for row in rows
        if str(row["id"]) not in chroma_ids
        or chroma_meta.get(str(row["id"]), {}).get("user_id") != row["user_id"]
        or chroma_meta.get(str(row["id"]), {}).get("status") != row["status"]
    ]
    if stale:
        collection.upsert(
            ids=[str(row["id"]) for row in stale],
            documents=[row["text"] for row in stale],
            metadatas=[_metadata(row) for row in stale],
        )
        repository.mark_synced([int(row["id"]) for row in stale])
    orphans = sorted(chroma_ids - sqlite_ids)
    if orphans:
        collection.delete(ids=orphans)
    return {"reindexed": len(stale), "orphans_deleted": len(orphans)}


def _lexical_overlap(left: str, right: str) -> float:
    a, b = set(_normalize(left)), set(_normalize(right))
    return len(a & b) / max(1, len(a | b))


def _find_duplicate(collection, text: str, user_id: str, memory_type: str) -> tuple[int | None, float | None]:
    rows = _repository().list_active_by_type(user_id, memory_type)
    normalized = _normalize(text)
    for row in rows:
        if _normalize(row["text"]) == normalized:
            return int(row["id"]), 0.0
    if not rows:
        return None, None
    try:
        result = collection.query(
            query_texts=[text], n_results=min(3, len(rows)),
            where={"$and": [{"user_id": {"$eq": user_id}}, {"type": {"$eq": memory_type}}]},
        )
        if result.get("ids") and result["ids"][0]:
            distance = float(result["distances"][0][0])
            candidate_id = int(result["ids"][0][0])
            candidate_text = next((row["text"] for row in rows if row["id"] == candidate_id), "")
            if distance <= DEDUP_DISTANCE_THRESHOLD and _lexical_overlap(text, candidate_text) >= 0.35:
                return candidate_id, distance
    except Exception:
        # 旧索引 metadata 不完整时仍可依赖文本精确去重。
        pass
    return None, None


def add_memory(
    text: str,
    type: str = "fact",
    importance: int = 5,
    *,
    user_id: str = DEFAULT_USER_ID,
    expires_at: dt.datetime | str | None = None,
    source_type: str = "inferred",
    confidence: float = 0.7,
) -> dict[str, Any]:
    """写入记忆；精确或高相似重复项只更新强度，不重复新增。"""
    text = str(text or "").strip()
    if not text:
        raise ValueError("记忆文本不能为空")
    importance = max(1, min(10, int(importance)))
    confidence = max(0.0, min(1.0, float(confidence)))
    if isinstance(expires_at, str):
        expires_at = dt.datetime.fromisoformat(expires_at)
    collection = _collection(create=True)
    duplicate_id, distance = _find_duplicate(collection, text, user_id, type)
    now = _now()
    if duplicate_id is not None:
        _repository().update_duplicate(duplicate_id, importance, confidence, now)
        print(f"--- 相似记忆已存在，更新 ID: {duplicate_id} ---")
        return {"id": duplicate_id, "action": "updated_duplicate", "distance": distance}

    repository = _repository()
    memory_id = repository.insert({
        "text": text, "type": type, "importance": importance,
        "created_at": now, "last_accessed_at": now, "user_id": user_id,
        "status": "pending", "expires_at": expires_at, "updated_at": now,
        "access_count": 0, "sync_status": "pending",
        "source_type": source_type, "confidence": confidence,
    })
    try:
        row = {
            "id": memory_id, "type": type, "importance": importance,
            "user_id": user_id, "status": "active", "source_type": source_type,
            "confidence": confidence,
        }
        collection.upsert(ids=[str(memory_id)], documents=[text], metadatas=[_metadata(row)])
        repository.mark_synced([memory_id])
    except Exception:
        repository.delete(memory_id)
        raise
    print(f"记忆已存入，ID: {memory_id}")
    return {"id": memory_id, "action": "created"}


def retrieve_memories(query_text: str, top_k: int = 3, *, user_id: str = DEFAULT_USER_ID) -> list[dict]:
    """召回当前用户未过期的记忆，弱化单纯访问热度的反馈循环。"""
    print(f"--- 检索与 '{query_text[:20]}...' 相关的长期记忆 ---")
    repository = _repository()
    active_count = repository.active_count(user_id, _now())
    if not active_count:
        return []
    collection = _collection(create=True)
    try:
        results = collection.query(
            query_texts=[query_text], n_results=min(top_k * 3, active_count),
            where={"$and": [{"user_id": {"$eq": user_id}}, {"status": {"$eq": "active"}}]},
        )
    except Exception:
        repair_memory_index(collection)
        results = collection.query(query_texts=[query_text], n_results=min(top_k * 3, active_count))
    if not results.get("ids") or not results["ids"][0]:
        return []

    ids = [int(value) for value in results["ids"][0]]
    rows = repository.get_active_by_ids(ids, user_id, _now())
    ranked = []
    now = _now()
    for memory_id, distance in zip(ids, results["distances"][0]):
        row = rows.get(memory_id)
        if not row:
            continue
        semantic = 1.0 / (1.0 + max(0.0, float(distance)))
        updated_value = row["updated_at"] or row["created_at"]
        updated = updated_value if isinstance(updated_value, dt.datetime) else dt.datetime.fromisoformat(str(updated_value))
        days = max(0.0, (now - updated).total_seconds() / 86400)
        recency = 1.0 / (1.0 + math.log1p(days))
        score = semantic * (0.7 + 0.03 * row["importance"]) * (0.9 + 0.1 * recency) * (0.8 + 0.2 * row["confidence"])
        ranked.append({
            "id": row["id"], "text": row["text"], "type": row["type"],
            "importance": row["importance"], "confidence": row["confidence"],
            "expires_at": row["expires_at"], "score": score,
        })
    top = sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
    if top:
        repository.touch([int(item["id"]) for item in top], now)
        print(f"检索到的Top-{len(top)} 记忆ID: {[item['id'] for item in top]}")
    return top


def delete_memory(memory_id: int, *, user_id: str = DEFAULT_USER_ID) -> bool:
    """先删向量，成功后再删 SQLite，避免留下不可检索的幽灵记录。"""
    repository = _repository()
    row = repository.get(memory_id, user_id)
    if not row:
        return False
    collection = _collection(create=True)
    collection.delete(ids=[str(memory_id)])
    return repository.delete(memory_id, user_id)


def view_memories(limit: int = 10, *, user_id: str = DEFAULT_USER_ID, include_inactive: bool = False) -> list[dict]:
    return _repository().view(limit, user_id, include_inactive, _now())


def expire_memories() -> int:
    """将到期记忆标记为 expired 并从向量索引删除。"""
    now = _now()
    repository = _repository()
    ids = repository.find_expired_ids(now)
    if not ids:
        return 0
    _collection(create=True).delete(ids=[str(value) for value in ids])
    repository.mark_expired(ids, now)
    return len(ids)
