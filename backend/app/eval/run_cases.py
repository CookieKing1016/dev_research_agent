from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

from app.agent.workflow import run_research_workflow


async def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    result = await run_research_workflow(
        query=case["query"],
        task_type=case.get("task_type", "research_report"),
        sources=case.get("sources", []),
        emit=emit,
    )
    return {
        "id": case["id"],
        "task_type": case.get("task_type", "research_report"),
        "query": case["query"],
        **result["eval"],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    rows = [await _run_case(case) for case in cases]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} eval rows to {out}")


if __name__ == "__main__":
    asyncio.run(main())
