from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.workflow import run_research_workflow
from app.storage.artifacts import ArtifactStore
from app.storage.db import Database


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = APP_DIR.parent
DB_PATH = APP_DIR / "devresearch.sqlite3"
SESSIONS_DIR = PROJECT_DIR / "sessions"

app = FastAPI(title="DevResearch Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(DB_PATH)
artifacts = ArtifactStore(SESSIONS_DIR)


class TaskCreate(BaseModel):
    query: str = Field(min_length=4)
    task_type: str = "research_report"
    sources: list[str] = Field(default_factory=list)


@app.on_event("startup")
def startup() -> None:
    db.init()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tasks")
async def create_task(payload: TaskCreate) -> dict[str, Any]:
    task_id = db.create_task(payload.query, payload.task_type)
    events: list[dict[str, Any]] = []

    async def emit(event: dict[str, Any]) -> None:
        db.add_trace(task_id, event["stage"], event["message"], event.get("data", {}))
        events.append(event)

    result = await run_research_workflow(
        query=payload.query,
        task_type=payload.task_type,
        sources=payload.sources,
        emit=emit,
    )
    db.finish_task(task_id, result)
    artifact_list = artifacts.save_run(
        task_id=task_id,
        query=payload.query,
        task_type=payload.task_type,
        sources=payload.sources,
        trace=events,
        result=result,
    )
    return {"task_id": task_id, "result": result, "trace": events, "artifacts": artifact_list}


@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict[str, Any]:
    return db.get_task(task_id)


@app.get("/tasks/{task_id}/trace")
def get_trace(task_id: int) -> list[dict[str, Any]]:
    return db.get_trace(task_id)


@app.get("/tasks/{task_id}/artifacts")
def list_artifacts(task_id: int) -> list[dict[str, str]]:
    return artifacts.list_artifacts(task_id)


@app.get("/tasks/{task_id}/artifacts/{filename}")
def download_artifact(task_id: int, filename: str) -> FileResponse:
    path = artifacts.resolve_artifact(task_id, filename)
    return FileResponse(path, filename=filename)


@app.websocket("/ws/run")
async def run_task_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        query = payload.get("query", "")
        task_type = payload.get("task_type", "research_report")
        sources = payload.get("sources", [])
        task_id = db.create_task(query, task_type)

        async def emit(event: dict[str, Any]) -> None:
            db.add_trace(task_id, event["stage"], event["message"], event.get("data", {}))
            await websocket.send_json({"task_id": task_id, **event})
            await asyncio.sleep(0)

        result = await run_research_workflow(query, task_type, sources, emit)
        db.finish_task(task_id, result)
        artifact_list = artifacts.save_run(
            task_id=task_id,
            query=query,
            task_type=task_type,
            sources=sources,
            trace=db.get_trace(task_id),
            result=result,
        )
        await websocket.send_json(
            {"task_id": task_id, "stage": "done", "result": result, "artifacts": artifact_list}
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:  # pragma: no cover - surfaced to UI during demo
        await websocket.send_json({"stage": "error", "message": str(exc)})
