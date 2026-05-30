from __future__ import annotations

from typing import Any, TypedDict


class Evidence(TypedDict):
    source: str
    title: str
    content: str
    score: float


class ResearchState(TypedDict, total=False):
    query: str
    task_type: str
    sources: list[str]
    plan: list[str]
    evidence: list[Evidence]
    analysis: dict[str, Any]
    draft: str
    critique: dict[str, Any]
    final_report: str
    eval: dict[str, Any]
    replan_count: int
