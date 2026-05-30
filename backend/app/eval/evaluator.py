from __future__ import annotations

import re
from typing import Any

from app.llm.client import LLMClient


SECTION_GROUPS = [
    ("task", ["task", "user request", "question"]),
    ("analysis", ["architecture", "analysis", "key findings", "findings"]),
    ("evidence", ["evidence", "citation", "source", "references"]),
    ("reproduction", ["reproduction", "reading path", "run path", "setup"]),
    ("improvement", ["improvement", "resume-ready", "next steps", "risks"]),
]

CITATION_RE = re.compile(r"\[(\d+)\]")


async def score_report_with_judge(
    query: str,
    report: str,
    evidence: list[dict[str, Any]],
    critique: dict[str, Any],
) -> dict[str, Any]:
    llm = LLMClient()
    if llm.enabled:
        attempt = await llm_judge_report(query, report, evidence, critique, llm)
        if attempt["ok"]:
            return {**attempt["score"], "judge_mode": "llm"}
        fallback = score_report(query, report, evidence, critique)
        return {**fallback, "judge_mode": "rule", "judge_error": attempt["error"]}
    return {**score_report(query, report, evidence, critique), "judge_mode": "rule"}


def score_report(
    query: str,
    report: str,
    evidence: list[dict[str, Any]],
    critique: dict[str, Any],
) -> dict[str, Any]:
    cited_indices = citation_indices(report)
    tool_call_accuracy = score_tool_call(evidence)
    citation_support = score_citation_support(cited_indices, evidence)
    report_completeness = score_report_completeness(report)
    factual_consistency = score_factual_consistency(critique, citation_support)
    total = (
        tool_call_accuracy * 0.2
        + citation_support * 0.3
        + report_completeness * 0.25
        + factual_consistency * 0.25
    )
    return {
        "tool_call_accuracy": round(tool_call_accuracy, 2),
        "citation_support": round(citation_support, 2),
        "report_completeness": round(report_completeness, 2),
        "factual_consistency": round(factual_consistency, 2),
        "total_score": round(total * 5, 2),
        "badcase": build_badcase(
            tool_call_accuracy=tool_call_accuracy,
            citation_support=citation_support,
            report_completeness=report_completeness,
            factual_consistency=factual_consistency,
        ),
        "query_preview": query[:80],
    }


async def llm_judge_report(
    query: str,
    report: str,
    evidence: list[dict[str, Any]],
    critique: dict[str, Any],
    llm: LLMClient,
) -> dict[str, Any]:
    evidence_text = "\n\n".join(
        f"[{idx + 1}] {item.get('title', '')}\nSource: {item.get('source', '')}\nContent: {item.get('content', '')[:1200]}"
        for idx, item in enumerate(evidence[:6])
    )
    system = (
        "You are EvalAgent, an impartial LLM-as-Judge for engineering research reports. "
        "Return only JSON. Scores must be floats from 0 to 1."
    )
    user = f"""Evaluate this report against the task and evidence.

User task:
{query}

Evidence:
{evidence_text}

Critic result:
{critique}

Report:
{report[:8000]}

Return JSON with exactly these fields:
{{
  "tool_call_accuracy": 0.0,
  "citation_support": 0.0,
  "report_completeness": 0.0,
  "factual_consistency": 0.0,
  "badcase": ["short issue description"]
}}
"""
    try:
        payload = await llm.json_chat(system, user, temperature=0.0, max_tokens=900)
        return {"ok": True, "score": normalize_judge_payload(query, payload), "error": None}
    except Exception as exc:
        return {"ok": False, "score": None, "error": compact_error(exc)}


def normalize_judge_payload(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    tool_call_accuracy = clamp01(payload.get("tool_call_accuracy"))
    citation_support = clamp01(payload.get("citation_support"))
    report_completeness = clamp01(payload.get("report_completeness"))
    factual_consistency = clamp01(payload.get("factual_consistency"))
    total = (
        tool_call_accuracy * 0.2
        + citation_support * 0.3
        + report_completeness * 0.25
        + factual_consistency * 0.25
    )
    return {
        "tool_call_accuracy": round(tool_call_accuracy, 2),
        "citation_support": round(citation_support, 2),
        "report_completeness": round(report_completeness, 2),
        "factual_consistency": round(factual_consistency, 2),
        "total_score": round(total * 5, 2),
        "badcase": string_list(payload.get("badcase")),
        "query_preview": query[:80],
    }


def score_tool_call(evidence: list[dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    if len(evidence) >= 3:
        return 1.0
    return 0.6 + 0.2 * len(evidence)


def score_citation_support(cited_indices: set[int], evidence: list[dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    valid_citations = {idx for idx in cited_indices if 1 <= idx <= len(evidence)}
    return min(len(valid_citations) / min(len(evidence), 4), 1.0)


def score_report_completeness(report: str) -> float:
    normalized = normalize(report)
    hits = 0
    for _name, markers in SECTION_GROUPS:
        if any(marker in normalized for marker in markers):
            hits += 1
    return hits / len(SECTION_GROUPS)


def score_factual_consistency(critique: dict[str, Any], citation_support: float) -> float:
    if critique.get("passed") and citation_support >= 0.5:
        return 1.0
    if citation_support >= 0.5:
        return 0.75
    return 0.45


def citation_indices(report: str) -> set[int]:
    return {int(match) for match in CITATION_RE.findall(report)}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def build_badcase(
    *,
    tool_call_accuracy: float,
    citation_support: float,
    report_completeness: float,
    factual_consistency: float,
) -> list[str]:
    issues = []
    if tool_call_accuracy < 1:
        issues.append("retrieval returned too few evidence items")
    if citation_support < 0.75:
        issues.append("report citations do not cover enough retrieved evidence")
    if report_completeness < 0.8:
        issues.append("report is missing expected sections")
    if factual_consistency < 0.75:
        issues.append("critic did not fully pass factual consistency")
    return issues


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def compact_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
