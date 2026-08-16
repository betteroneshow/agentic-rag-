# -*- coding: utf-8 -*-
"""Agentic RAG 的 Streamlit 对话界面。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st

# 所有数据库路径均为项目相对路径，确保从任意目录启动时行为一致。
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from agentic_rag import memory  # noqa: E402
from agentic_rag.graph import build_graph  # noqa: E402
from config import LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, RERANK_MODEL_NAME  # noqa: E402

PERSIST_PATH = BASE_DIR / "chroma_db"
GRAPH_CONFIG = {
    "configurable": {"thread_id": "streamlit-session"},
    "recursion_limit": 50,
}


@st.cache_resource
def get_graph():
    """图结构和模型客户端跨 Streamlit 重绘复用。"""
    memory.initialize_memory_db()
    return build_graph()


@st.cache_resource
def get_collection_counts() -> dict[str, int]:
    if not PERSIST_PATH.exists():
        return {}
    client = chromadb.PersistentClient(path=str(PERSIST_PATH))
    counts = {}
    for collection in client.list_collections():
        counts[collection.name] = collection.count()
    return counts


def initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("user_id", os.getenv("AGENT_USER_ID", memory.DEFAULT_USER_ID))
    st.session_state.setdefault(
        "agent_session",
        {
            "user_id": st.session_state.user_id,
            "conversation_history": [],
            "conversation_summary": "",
            "conversation_turn_count": 0,
            "working_memory": {},
        },
    )


def reset_session() -> None:
    """只清空当前短期会话，不删除长期记忆。"""
    st.session_state.messages = []
    st.session_state.agent_session = {
        "user_id": st.session_state.user_id,
        "conversation_history": [],
        "conversation_summary": "",
        "conversation_turn_count": 0,
        "working_memory": {},
    }


def normalize_documents(documents: list[Any]) -> list[dict]:
    """将本地 Document 和网络搜索字典统一为可展示来源。"""
    sources = []
    seen = set()
    for index, document in enumerate(documents or [], start=1):
        if hasattr(document, "page_content"):
            content = str(document.page_content or "")
            metadata = dict(getattr(document, "metadata", {}) or {})
        elif isinstance(document, dict):
            content = str(
                document.get("content") or document.get("snippet")
                or document.get("text") or document
            )
            metadata = dict(document.get("metadata") or {})
            for key in ("title", "url", "source"):
                if document.get(key):
                    metadata.setdefault(key, document[key])
        else:
            content, metadata = str(document), {}

        label = (
            metadata.get("title") or metadata.get("attraction")
            or metadata.get("destination") or metadata.get("source")
            or metadata.get("url") or f"来源 {index}"
        )
        identity = (str(label), content[:160])
        if identity in seen:
            continue
        seen.add(identity)
        sources.append({"label": str(label), "content": content, "metadata": metadata})
    return sources


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"查看检索依据（{len(sources)} 条）"):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"**{index}. {source['label']}**")
            url = source["metadata"].get("url")
            if url:
                st.markdown(f"[打开原始网页]({url})")
            st.caption(source["content"][:1200] + ("…" if len(source["content"]) > 1200 else ""))
            if index < len(sources):
                st.divider()


def render_trace(trace: dict) -> None:
    with st.expander("查看 Agent 执行状态"):
        col1, col2, col3 = st.columns(3)
        col1.metric("检索路由", trace.get("route") or "direct")
        col2.metric("对话轮次", trace.get("conversation_turn_count", 0))
        col3.metric("修正次数", trace.get("correction_attempts", 0))
        completed = trace.get("completed_steps") or []
        if completed:
            st.markdown("**已完成：** " + " → ".join(map(str, completed)))
        if trace.get("updated_query"):
            st.markdown(f"**改写查询：** {trace['updated_query']}")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🧭 Agentic RAG")
        st.caption("旅游知识问答与行程规划助手")
        if st.button("＋ 新建会话", use_container_width=True):
            reset_session()
            st.rerun()

        st.divider()
        st.markdown("#### 当前会话")
        agent_session = st.session_state.agent_session
        st.metric("对话轮次", agent_session.get("conversation_turn_count", 0))
        summary = agent_session.get("conversation_summary")
        if summary:
            with st.expander("短期记忆摘要"):
                st.write(summary)
        working = agent_session.get("working_memory") or {}
        if working and working.get("goal"):
            with st.expander("工作记忆", expanded=True):
                st.markdown(f"**当前目标：** {working.get('goal')}")
                st.caption(f"阶段：{working.get('current_stage', '-')} · 状态：{working.get('status', '-')}")
                if working.get("origin") or working.get("destination"):
                    st.markdown(f"**路线：** {working.get('origin') or '待补充'} → {working.get('destination') or '待补充'}")
                if working.get("constraints"):
                    st.markdown("**约束：** " + "；".join(working["constraints"]))
                if working.get("pending_items"):
                    st.markdown("**待办：** " + "；".join(working["pending_items"]))
                if working.get("missing_information"):
                    st.warning("待补充：" + "、".join(working["missing_information"]))

        st.markdown("#### 知识库")
        try:
            counts = get_collection_counts()
            st.caption(f"父块：{counts.get('doc_chunks', 0):,}")
            st.caption(f"检索子块：{counts.get('retrieval_chunks', 0):,}")
            st.caption(f"文档摘要：{counts.get('doc_summaries', 0):,}")
        except Exception as exc:
            st.warning(f"无法读取知识库：{exc}")

        with st.expander("最近长期记忆"):
            try:
                memories = memory.view_memories(limit=8, user_id=st.session_state.user_id)
                if not memories:
                    st.caption("暂无长期记忆")
                for item in memories:
                    st.markdown(f"- {item['text']}")
            except Exception as exc:
                st.caption(f"读取失败：{exc}")

        st.divider()
        st.caption(f"LLM · {LLM_MODEL_NAME}")
        st.caption(f"Embedding · {EMBEDDING_MODEL_NAME}")
        st.caption(f"Rerank · {RERANK_MODEL_NAME}")


st.set_page_config(
    page_title="Agentic RAG 旅游助手",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1050px; padding-top: 2rem;}
    [data-testid="stChatMessage"] {border: 1px solid rgba(128,128,128,.16); border-radius: 16px; padding: .35rem .7rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
    </style>
    """,
    unsafe_allow_html=True,
)

initialize_session()
render_sidebar()

st.title("和旅游 Agent 对话")
st.caption("支持本地知识库混合检索、实时网络搜索、上下文压缩及长短期记忆。")

if not st.session_state.messages:
    st.info("例如：我准备去长沙玩三天，预算 3000 元，帮我安排一个交通省心的行程。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            if message.get("trace"):
                render_trace(message["trace"])

prompt = st.chat_input("输入旅行问题或指令……")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Agent 正在分析问题…", expanded=True)
        try:
            status.write("检索相关长期记忆")
            graph = get_graph()
            inputs = {"query": prompt, **st.session_state.agent_session}
            final_state = graph.invoke(inputs, config=GRAPH_CONFIG)
            status.write("完成检索、精排和答案校验")
            status.update(label="处理完成", state="complete", expanded=False)

            answer = final_state.get("response") or "抱歉，本次没有生成有效答案。"
            sources = normalize_documents(final_state.get("documents", []))
            trace = {
                key: final_state.get(key)
                for key in (
                    "route", "updated_query", "conversation_turn_count",
                    "correction_attempts", "completed_steps", "current_parameters",
                )
            }
            st.markdown(answer)
            render_sources(sources)
            render_trace(trace)

            st.session_state.agent_session = {
                "user_id": final_state.get("user_id", st.session_state.user_id),
                "conversation_history": final_state.get("conversation_history", []),
                "conversation_summary": final_state.get("conversation_summary", ""),
                "conversation_turn_count": final_state.get("conversation_turn_count", 0),
                "working_memory": final_state.get("working_memory", {}),
            }
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "trace": trace}
            )
        except Exception as exc:
            status.update(label="处理失败", state="error", expanded=True)
            st.error(f"Agent 执行失败：{exc}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Agent 执行失败：{exc}", "sources": []}
            )
