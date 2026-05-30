from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agent.state import ResearchState
from app.eval.evaluator import score_report
from app.retrieval.hybrid import HybridRetriever
from app.tools.github_tool import GitHubTool

Emit = Callable[[dict[str, Any]], Awaitable[None]]  # 异步事件发送函数
Route = Literal["replan", "eval_result"]

#对外暴露的主入口，负责接收用户输入并运行图。
async def run_research_workflow(
    query: str,
    task_type: str,
    sources: list[str],
    emit: Emit, #异步事件发送函数
) -> dict[str, Any]:
    initial_state: ResearchState = {
        "query": query,
        "task_type": task_type,
        "sources": sources,
        "replan_count": 0,
    }

    graph = build_research_graph(emit)
    state = await graph.ainvoke(initial_state)

    return {
        "query": state["query"],
        "task_type": state["task_type"],
        "plan": state["plan"],
        "evidence": state["evidence"],
        "analysis": state["analysis"],
        "critique": state["critique"],
        "eval": state["eval"],
        "report": state["final_report"],
    }

#负责搭建 LangGraph 工作流，包括节点和边。
def build_research_graph(emit: Emit):
    workflow = StateGraph(ResearchState)

    async def plan_node(state: ResearchState) -> ResearchState:
        await plan_task(state, emit)
        return state

    async def retrieve_node(state: ResearchState) -> ResearchState:
        await retrieve_context(state, emit)
        return state

    async def analyze_node(state: ResearchState) -> ResearchState:
        await analyze_code_or_docs(state, emit)
        return state

    async def draft_node(state: ResearchState) -> ResearchState:
        await draft_report(state, emit)
        return state

    async def critic_node(state: ResearchState) -> ResearchState:
        await critic_review(state, emit)
        return state

    async def replan_node(state: ResearchState) -> ResearchState:
        await replan(state, emit)
        return state

    async def eval_node(state: ResearchState) -> ResearchState:
        await eval_result(state, emit)
        return state

    workflow.add_node("plan_task", plan_node)
    workflow.add_node("retrieve_context", retrieve_node)
    workflow.add_node("analyze_code_or_docs", analyze_node)
    workflow.add_node("draft_report", draft_node)
    workflow.add_node("critic_review", critic_node)
    workflow.add_node("replan", replan_node)
    workflow.add_node("eval_result", eval_node)

    workflow.add_edge(START, "plan_task")
    workflow.add_edge("plan_task", "retrieve_context")
    workflow.add_edge("retrieve_context", "analyze_code_or_docs")
    workflow.add_edge("analyze_code_or_docs", "draft_report")
    workflow.add_edge("draft_report", "critic_review")
    workflow.add_conditional_edges(
        "critic_review",
        route_after_critic,
        {"replan": "replan", "eval_result": "eval_result"},
    )
    workflow.add_edge("replan", "retrieve_context")
    workflow.add_edge("eval_result", END)

    return workflow.compile()


def route_after_critic(state: ResearchState) -> Route:
    if should_replan(state):
        return "replan"
    return "eval_result"


async def plan_task(state: ResearchState, emit: Emit) -> None:
    query = state["query"]
    plan = [
        "Identify task type and expected deliverable",
        "Retrieve GitHub, document, and local knowledge evidence",
        "Analyze code or technical material flow",
        "Generate a technical report with citations",
        "Run Critic checks for factual support, omissions, and risk",
        "Output evaluation score and badcase suggestions",
    ]
    if "issue" in query.lower() or "bug" in query.lower():
        plan.insert(2, "Locate relevant files, entrypoints, and risk points")
    state["plan"] = plan
    await emit({"stage": "plan_task", "message": "PlannerAgent generated the task plan", "data": {"plan": plan}})


async def retrieve_context(state: ResearchState, emit: Emit) -> None:
    github = GitHubTool()
    documents = github.load_seed_documents(state["query"], state.get("sources", []))
    retriever = HybridRetriever(documents)
    evidence = retriever.search(state["query"], top_k=6)
    state["evidence"] = evidence
    await emit(
        {
            "stage": "retrieve_context",
            "message": f"SearchAgent retrieved {len(evidence)} evidence items",
            "data": {"evidence": evidence},
        }
    )


async def analyze_code_or_docs(state: ResearchState, emit: Emit) -> None:
    evidence = state.get("evidence", [])
    keywords = sorted({word for item in evidence for word in item["content"].split() if len(word) > 5})[:12]
    analysis = {
        "architecture": "Planner -> Retriever -> Tool Caller -> Writer -> Critic -> Evaluator",
        "matched_keywords": keywords,
        "risk_points": [
            "The agent may over-generate when evidence is weak",
            "External tool failures can reduce report completeness",
            "Large repositories need layered indexing",
        ],
    }
    state["analysis"] = analysis
    await emit({"stage": "analyze_code_or_docs", "message": "CodeAgent completed structure analysis", "data": analysis})


async def draft_report(state: ResearchState, emit: Emit) -> None:
    evidence = state.get("evidence", [])
    citations = "\n".join(f"- [{i + 1}] {item['title']}: {item['source']}" for i, item in enumerate(evidence[:4]))
    report = f"""# Engineering Task Analysis Report

## Task
{state['query']}

## Execution Plan
The system uses multi-agent collaboration: PlannerAgent decomposes the task,
SearchAgent retrieves evidence, CodeAgent analyzes code or document flow,
WriterAgent drafts the report, CriticAgent verifies factual support, and
EvalAgent outputs quality scores.

## Key Findings
1. This task is suitable for a stateful Agent workflow based on LangGraph or AgentScope.
2. The technical chain should cover tool calling, RAG retrieval, Critic review, and Agentic Eval.
3. The resume version should emphasize traceability, replan loop, and evaluation metrics.

## Evidence
{citations}
"""
    state["draft"] = report
    state["final_report"] = report
    await emit({"stage": "draft_report", "message": "WriterAgent generated the report draft", "data": {"chars": len(report)}})


async def critic_review(state: ResearchState, emit: Emit) -> None:
    evidence_count = len(state.get("evidence", []))
    has_citations = "[1]" in state.get("draft", "")
    needs_replan = evidence_count < 3 or not has_citations
    critique = {
        "passed": not needs_replan,
        "needs_replan": needs_replan,
        "issues": [] if not needs_replan else ["Evidence is insufficient or the report lacks citations"],
        "suggestions": ["Add README, Issue, and code entrypoint evidence", "Export failure cases for tuning"],
    }
    state["critique"] = critique
    await emit({"stage": "critic_review", "message": "CriticAgent completed factual support checks", "data": critique})


def should_replan(state: ResearchState) -> bool:
    return bool(state.get("critique", {}).get("needs_replan")) and state.get("replan_count", 0) < 1


async def replan(state: ResearchState, emit: Emit) -> None:
    state["replan_count"] = state.get("replan_count", 0) + 1
    state["sources"] = state.get("sources", []) + ["local-agent-patterns", "eval-rubric"]
    await emit(
        {
            "stage": "replan",
            "message": "Evidence was insufficient, so the graph triggered one replan pass",
            "data": {"sources": state["sources"], "replan_count": state["replan_count"]},
        }
    )


async def eval_result(state: ResearchState, emit: Emit) -> None:
    result = score_report(
        query=state["query"],
        report=state.get("final_report", ""),
        evidence=state.get("evidence", []),
        critique=state.get("critique", {}),
    )
    state["eval"] = result
    await emit({"stage": "eval_result", "message": "EvalAgent produced the final score", "data": result})
