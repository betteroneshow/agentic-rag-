# -*- coding: utf-8 -*-
"""ChromaDB 本地可视化管理界面。"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PERSIST_PATH = BASE_DIR / "chroma_db"
MANIFEST_PATH = PERSIST_PATH / "ingestion_manifest.json"
FILTER_FIELDS = [
    "source_file", "source", "destination", "full_destination",
    "attraction", "doc_type", "record_level", "input_type",
    "ocr_model", "extraction_method",
]


@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=str(PERSIST_PATH))


@st.cache_resource
def get_embedding_function():
    # 仅在语义搜索时加载，普通浏览不会调用外部 Embedding API。
    from agentic_rag.chains import get_embedding_function as create_embedding
    return create_embedding()


def build_where(filters: dict[str, str]) -> dict | None:
    clauses = [
        {field: {"$eq": value.strip()}}
        for field, value in filters.items() if value.strip()
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}


def invalidate_manifest_sources(source_files: set[str]) -> None:
    """删除向量后让对应文件在下次 ingest 时重新入库。"""
    if not source_files or not MANIFEST_PATH.exists():
        return
    manifest = load_manifest()
    files = manifest.setdefault("files", {})
    changed = False
    for source_file in source_files:
        if source_file in files:
            del files[source_file]
            changed = True
    if changed:
        temp_path = MANIFEST_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(MANIFEST_PATH)


def normalize_results(results: dict, semantic: bool = False) -> list[dict]:
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", []) if semantic else []
    if semantic:
        ids = ids[0] if ids else []
        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []
    return [
        {
            "id": item_id,
            "document": documents[index] if index < len(documents) else "",
            "metadata": metadatas[index] if index < len(metadatas) else {},
            "distance": distances[index] if semantic and index < len(distances) else None,
        }
        for index, item_id in enumerate(ids)
    ]


def render_record(record: dict, number: int) -> None:
    metadata = record["metadata"] or {}
    title = metadata.get("attraction") or metadata.get("destination") or record["id"]
    distance = record.get("distance")
    suffix = f" · 距离 {distance:.4f}" if distance is not None else ""
    with st.expander(f"{number}. {title}{suffix}"):
        st.caption(f"ID：{record['id']}")
        st.json(metadata)
        st.text_area(
            "文档内容", value=record["document"] or "", height=240,
            disabled=True, key=f"content-{record['id']}-{number}",
        )


st.set_page_config(page_title="向量数据库管理", page_icon="🧭", layout="wide")
st.title("🧭 ChromaDB 向量数据库管理")
st.caption(f"数据库位置：{PERSIST_PATH}")

if not PERSIST_PATH.exists():
    st.error("向量数据库目录不存在，请先运行 python ingest.py。")
    st.stop()

client = get_client()
collections = sorted(client.list_collections(), key=lambda item: item.name)
if not collections:
    st.warning("数据库中还没有集合。")
    st.stop()

collection_names = [item.name for item in collections]
manifest = load_manifest()
overview_tab, browse_tab, manage_tab = st.tabs(["集合概览", "浏览与搜索", "数据管理"])

with overview_tab:
    columns = st.columns(max(1, min(4, len(collections) + 1)))
    for index, item in enumerate(collections):
        columns[index % len(columns)].metric(item.name, item.count())
    columns[-1].metric("增量清单文件", len(manifest.get("files", {})))
    st.dataframe(
        [{"集合": item.name, "记录数": item.count()} for item in collections],
        width="stretch", hide_index=True,
    )
    if manifest.get("files"):
        with st.expander("查看已完成入库的源文件"):
            st.dataframe(
                [{"文件": path, "指纹": value} for path, value in manifest["files"].items()],
                width="stretch", hide_index=True,
            )

with browse_tab:
    selected_collection = st.selectbox("集合", collection_names, key="browse_collection")
    collection = client.get_collection(selected_collection)
    search_mode = st.radio("查询方式", ["分页浏览", "关键词包含", "语义搜索"], horizontal=True)
    with st.expander("Metadata 精确过滤"):
        filter_columns = st.columns(3)
        filters = {
            field: filter_columns[index % 3].text_input(field, key=f"filter-{field}")
            for index, field in enumerate(FILTER_FIELDS)
        }
    where = build_where(filters)
    rows: list[dict] = []
    try:
        if search_mode == "分页浏览":
            page_size = st.selectbox("每页数量", [10, 20, 50, 100], index=1)
            total = collection.count()
            max_page = max(1, (total + page_size - 1) // page_size)
            page = st.number_input("页码", min_value=1, max_value=max_page, value=1)
            kwargs = {"limit": page_size, "offset": (page - 1) * page_size,
                      "include": ["documents", "metadatas"]}
            if where:
                kwargs["where"] = where
            rows = normalize_results(collection.get(**kwargs))
            st.caption(f"集合共 {total} 条；当前显示第 {page}/{max_page} 页")
        elif search_mode == "关键词包含":
            keyword = st.text_input("关键词")
            limit = st.slider("返回数量", 1, 100, 20)
            if keyword:
                kwargs = {"where_document": {"$contains": keyword}, "limit": limit,
                          "include": ["documents", "metadatas"]}
                if where:
                    kwargs["where"] = where
                rows = normalize_results(collection.get(**kwargs))
        else:
            query = st.text_input("语义查询")
            top_k = st.slider("Top K", 1, 50, 10)
            if query:
                semantic_collection = client.get_collection(
                    selected_collection, embedding_function=get_embedding_function()
                )
                kwargs = {"query_texts": [query],
                          "n_results": min(top_k, max(1, semantic_collection.count())),
                          "include": ["documents", "metadatas", "distances"]}
                if where:
                    kwargs["where"] = where
                rows = normalize_results(semantic_collection.query(**kwargs), semantic=True)
    except Exception as exc:
        st.error(f"查询失败：{exc}")

    st.session_state["visible_records"] = rows
    st.subheader(f"查询结果（{len(rows)}）")
    for index, record in enumerate(rows, start=1):
        render_record(record, index)

with manage_tab:
    st.warning("删除不可撤销。对应源文件会从增量清单移除，下次 ingest 将重新处理。")
    manage_collection_name = st.selectbox("要管理的集合", collection_names, key="manage_collection")
    visible_records = st.session_state.get("visible_records", [])
    visible_ids = [record["id"] for record in visible_records]
    if not visible_ids:
        st.info("请先在“浏览与搜索”页查询出要管理的记录。")
    selected_ids = st.multiselect("选择要删除的记录 ID", visible_ids)
    confirmation = st.text_input("输入 DELETE 确认删除", type="password")
    if st.button("删除选中记录", type="primary", disabled=not selected_ids):
        if confirmation != "DELETE":
            st.error("确认词不正确，未执行删除。")
        else:
            manage_collection = client.get_collection(manage_collection_name)
            existing = manage_collection.get(ids=selected_ids, include=["metadatas"])
            source_files = {
                metadata.get("source_file")
                for metadata in existing.get("metadatas", [])
                if metadata and metadata.get("source_file")
            }
            manage_collection.delete(ids=selected_ids)
            invalidate_manifest_sources(source_files)
            st.success(f"已删除 {len(selected_ids)} 条记录。")
            st.session_state["visible_records"] = []
            st.rerun()
