# -*- coding: utf-8 -*-
"""
@desc: 构建并编译Agentic RAG的工作流图
"""

from langgraph.graph import StateGraph, END

from agentic_rag.state import AgentState
from agentic_rag.nodes import (
    retrieve_memory_node,
    consolidate_memory_node,
    route_query_node,
    rewrite_query_node,
    retrieve_documents_node, # new
    grade_documents_node,    # new
    generate_response_node,
    grade_relevance_node,
    direct_response_node,
    retrieval_fallback_node,
    check_route_requirements_node,
    request_route_clarification_node,
    update_working_memory_node,
)

def build_graph():
    """构建并返回集成了“自省”能力的、包含内外双循环的LangGraph图。"""
    workflow = StateGraph(AgentState)

    # --- 添加所有节点 ---
    # 记忆相关
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("consolidate_memory", consolidate_memory_node)
    # 核心流程
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("update_working_memory", update_working_memory_node)
    workflow.add_node("check_route_requirements", check_route_requirements_node)
    workflow.add_node("request_route_clarification", request_route_clarification_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("retrieval_fallback", retrieval_fallback_node)
    # 外部循环评估
    workflow.add_node("grade_relevance", grade_relevance_node)

    # --- 定义边 ---

    # 1. 从“回忆”开始
    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "update_working_memory")
    workflow.add_edge("update_working_memory", "check_route_requirements")
    workflow.add_conditional_edges(
        "check_route_requirements",
        lambda state: "clarify" if state.get("needs_clarification") else "continue",
        {
            "clarify": "request_route_clarification",
            "continue": "route_query",
        },
    )
    workflow.add_edge("request_route_clarification", END)

    # 2. “路由”后，对于需要检索的，先“重写查询”
    workflow.add_conditional_edges(
        "route_query",
        lambda state: state["route"],
        {
            "web_search": "rewrite_query",
            "hierarchical_search": "rewrite_query",
            "direct_chunk_search": "rewrite_query",
            "direct": "direct_response" # “直接回答”路由，跳过所有检索和生成
        }
    )
    
    # 3. “重写查询”后，开始“内循环”：检索->评估->决策
    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")

    # 4. “内循环”的核心决策
    def decide_after_document_grading(state: AgentState):
        """在评估文档后，决定是生成答案，还是切换策略重试。"""
        if state.get("documents_are_relevant"):
            print("---决策：文档相关，进入答案生成---")
            return "generate"

        if not state.get("retrieval_exhausted"):
            print(f"---决策：使用新策略 '{state['route']}' 重试---")
            return "retry_retrieve"

        print("---决策：所有检索策略均失败，无法找到相关文档---")
        return "fallback"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_document_grading,
        {
            "generate": "generate_response",
            "retry_retrieve": "retrieve_documents", # 回到检索节点，形成循环
            "fallback": "retrieval_fallback"
        }
    )

    workflow.add_edge("retrieval_fallback", "consolidate_memory")

    # 5. “外循环”：生成答案 -> 评估答案
    workflow.add_edge("generate_response", "grade_relevance")
    # “直接回答”也连接到最终评估
    workflow.add_edge("direct_response", "grade_relevance")

    # 6. “外循环”的决策
    def decide_after_answer_grading(state: AgentState):
        """在评估最终答案后，决定是结束还是重试。"""
        if state["is_relevant"]:
            print("---决策：答案相关，流程结束---")
            return "end"
        
        if state.get("correction_attempts", 0) >= 2:
            print("---决策：已达到最大重试次数，流程结束---")
            return "end"
        else:
            print("---决策：答案不相关，触发修正性重写---")
            return "retry"

    workflow.add_conditional_edges(
        "grade_relevance",
        decide_after_answer_grading,
        {
            "end": "consolidate_memory", # 答案相关或达到最大次数，去“复盘”并形成记忆
            "retry": "rewrite_query"     # 答案不相关，重写查询
        }
    )
    
    # 7. “复盘记忆”后，流程结束
    workflow.add_edge("consolidate_memory", END)

    # 编译图
    graph = workflow.compile()
    return graph
