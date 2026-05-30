from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.agent.state import ResearchState
from app.eval.evaluator import score_report_with_judge
from app.llm.client import LLMClient
from app.retrieval.hybrid import HybridRetriever
from app.tools.github_tool import GitHubTool

Emit = Callable[[dict[str, Any]], Awaitable[None]]
Route = Literal["replan", "eval_result"]


@dataclass
class LLMAttempt:
    value: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


async def run_research_workflow(
    query: str,
    task_type: str,
    sources: list[str],
    emit: Emit,
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
    return "replan" if should_replan(state) else "eval_result"


async def plan_task(state: ResearchState, emit: Emit) -> None:
    query = state["query"]
    llm = LLMClient()
    attempt = await llm_plan(query, state["task_type"], llm)
    plan = attempt.value or [
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
    if llm.enabled and attempt.error:
        await emit({"stage": "llm_error", "message": f"PlannerAgent LLM failed: {attempt.error}", "data": {"node": "plan_task"}})
    mode = "llm" if attempt.ok else "rule"
    await emit({"stage": "plan_task", "message": f"PlannerAgent generated the task plan ({mode})", "data": {"plan": plan}})


async def retrieve_context(state: ResearchState, emit: Emit) -> None:
    github = GitHubTool()
    documents = github.load_seed_documents(state["query"], state.get("sources", []))
    retriever = HybridRetriever(documents)
    evidence = retriever.search(state["query"], top_k=6)
    state["evidence"] = evidence
    await emit({"stage": "retrieve_context", "message": f"SearchAgent retrieved {len(evidence)} evidence items", "data": {"evidence": evidence}})


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
    llm = LLMClient()
    attempt = await llm_report(state, llm)
    report = attempt.value
    if llm.enabled and attempt.error:
        await emit({"stage": "llm_error", "message": f"WriterAgent LLM failed: {attempt.error}", "data": {"node": "draft_report"}})
    if not report:
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
    mode = "llm" if attempt.ok else "template"
    await emit({"stage": "draft_report", "message": f"WriterAgent generated the report draft ({mode})", "data": {"chars": len(report)}})


async def critic_review(state: ResearchState, emit: Emit) -> None:
    evidence_count = len(state.get("evidence", []))
    has_citations = "[1]" in state.get("draft", "")
    needs_replan = evidence_count < 3 or not has_citations
    llm = LLMClient()
    attempt = await llm_critique(state, llm)
    critique = attempt.value
    if llm.enabled and attempt.error:
        await emit({"stage": "llm_error", "message": f"CriticAgent LLM failed: {attempt.error}", "data": {"node": "critic_review"}})
    if not critique:
        critique = {
            "passed": not needs_replan,
            "needs_replan": needs_replan,
            "issues": [] if not needs_replan else ["Evidence is insufficient or the report lacks citations"],
            "suggestions": ["Add README, Issue, and code entrypoint evidence", "Export failure cases for tuning"],
        }
    state["critique"] = critique
    mode = "llm" if attempt.ok else "rule"
    await emit({"stage": "critic_review", "message": f"CriticAgent completed factual support checks ({mode})", "data": critique})


def should_replan(state: ResearchState) -> bool:
    return bool(state.get("critique", {}).get("needs_replan")) and state.get("replan_count", 0) < 1


async def replan(state: ResearchState, emit: Emit) -> None:
    state["replan_count"] = state.get("replan_count", 0) + 1
    state["sources"] = state.get("sources", []) + ["local-agent-patterns", "eval-rubric"]
    await emit({"stage": "replan", "message": "Evidence was insufficient, so the graph triggered one replan pass", "data": {"sources": state["sources"], "replan_count": state["replan_count"]}})


async def eval_result(state: ResearchState, emit: Emit) -> None:
    result = await score_report_with_judge(
        query=state["query"],
        report=state.get("final_report", ""),
        evidence=state.get("evidence", []),
        critique=state.get("critique", {}),
    )
    if result.get("judge_error"):
        await emit({"stage": "llm_error", "message": f"EvalAgent LLM judge failed: {result['judge_error']}", "data": {"node": "eval_result"}})
    state["eval"] = result
    await emit({"stage": "eval_result", "message": f"EvalAgent produced the final score ({result.get('judge_mode', 'rule')})", "data": result})


async def llm_plan(query: str, task_type: str, llm: LLMClient) -> LLMAttempt:
    if not llm.enabled:
        return LLMAttempt(error="LLM_API_KEY is not configured")
    system = "You are PlannerAgent. Return only JSON with a string array field named plan."
    user = f"""Task type: {task_type}
User query: {query}

Create 5-7 concrete steps for a multi-agent engineering research workflow.
The steps should mention retrieval, evidence analysis, report writing, critique, and evaluation.
"""
    try:
        payload = await llm.json_chat(system, user, max_tokens=700)
        plan = payload.get("plan")
        if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
            return LLMAttempt(value=plan)
        return LLMAttempt(error=f"Invalid planner JSON schema: {payload}")
    except Exception as exc:
        return LLMAttempt(error=compact_error(exc))


async def llm_report(state: ResearchState, llm: LLMClient) -> LLMAttempt:
    if not llm.enabled:
        return LLMAttempt(error="LLM_API_KEY is not configured")
    evidence = state.get("evidence", [])[:6]
    evidence_text = "\n\n".join(
        f"[{idx + 1}] {item['title']}\nSource: {item['source']}\nContent: {item['content'][:1800]}"
        for idx, item in enumerate(evidence)
    )
    system = (
        "You are WriterAgent. Write a concise Markdown engineering report. "
        "Use citations like [1], [2] whenever making factual claims. "
        "Do not invent facts beyond the provided evidence."
    )
    user = f"""User task:
{state['query']}

Plan:
{state.get('plan', [])}

Analysis:
{state.get('analysis', {})}

Evidence:
{evidence_text}

Write sections:
1. Task
2. Evidence-backed architecture notes
3. Reproduction or reading path
4. Resume-ready improvement ideas
5. Risks and next steps
"""
    try:
        return LLMAttempt(value=await llm.chat(system, user, temperature=0.2, max_tokens=1800))
    except Exception as exc:
        return LLMAttempt(error=compact_error(exc))


async def llm_critique(state: ResearchState, llm: LLMClient) -> LLMAttempt:
    if not llm.enabled:
        return LLMAttempt(error="LLM_API_KEY is not configured")
    system = (
        "You are CriticAgent. Return only JSON with fields: "
        "passed(boolean), needs_replan(boolean), issues(array of strings), suggestions(array of strings)."
    )
    user = f"""User task:
{state['query']}

Evidence count: {len(state.get('evidence', []))}
Report:
{state.get('draft', '')[:6000]}

Judge whether the report is supported by citations, answers the task, and has enough evidence.
Set needs_replan=true only if missing citations, weak evidence, or not answering the task.
"""
    try:
        payload = await llm.json_chat(system, user, max_tokens=800)
        return LLMAttempt(
            value={
                "passed": bool(payload.get("passed")),
                "needs_replan": bool(payload.get("needs_replan")),
                "issues": _string_list(payload.get("issues")),
                "suggestions": _string_list(payload.get("suggestions")),
            }
        )
    except Exception as exc:
        return LLMAttempt(error=compact_error(exc))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def compact_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
