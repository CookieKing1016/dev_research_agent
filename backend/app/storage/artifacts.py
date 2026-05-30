from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACT_FILENAMES = {
    "query.txt",
    "plan.json",
    "evidence.json",
    "trace.json",
    "report.md",
    "critique.json",
    "eval.json",
    "trajectory.jsonl",
}


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def task_dir(self, task_id: int) -> Path:
        return self.root / str(task_id)

    def save_run(
        self,
        *,
        task_id: int,
        query: str,
        task_type: str,
        sources: list[str],
        trace: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> list[dict[str, str]]:
        directory = self.task_dir(task_id)
        directory.mkdir(parents=True, exist_ok=True)

        self._write_text(directory / "query.txt", query)
        self._write_json(directory / "plan.json", result.get("plan", []))
        self._write_json(directory / "evidence.json", result.get("evidence", []))
        self._write_json(directory / "trace.json", trace)
        self._write_text(directory / "report.md", result.get("report", ""))
        self._write_json(directory / "critique.json", result.get("critique", {}))
        self._write_json(directory / "eval.json", result.get("eval", {}))
        self._write_jsonl(
            directory / "trajectory.jsonl",
            {
                "task_id": task_id,
                "query": query,
                "task_type": task_type,
                "sources": sources,
                "plan": result.get("plan", []),
                "evidence": result.get("evidence", []),
                "analysis": result.get("analysis", {}),
                "critique": result.get("critique", {}),
                "eval": result.get("eval", {}),
                "report": result.get("report", ""),
                "trace": trace,
                "human_label": None,
            },
        )

        return self.list_artifacts(task_id)

    def list_artifacts(self, task_id: int) -> list[dict[str, str]]:
        directory = self.task_dir(task_id)
        if not directory.exists():
            return []
        artifacts = []
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name in ARTIFACT_FILENAMES:
                artifacts.append({"name": path.name, "path": str(path)})
        return artifacts

    def resolve_artifact(self, task_id: int, filename: str) -> Path:
        if filename not in ARTIFACT_FILENAMES:
            raise ValueError("Unknown artifact")
        path = self.task_dir(task_id) / filename
        root = self.task_dir(task_id).resolve()
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise ValueError("Invalid artifact path")
        if not resolved.exists():
            raise FileNotFoundError(filename)
        return resolved

    def _write_text(self, path: Path, text: str) -> None:
        path.write_text(text, encoding="utf-8")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
