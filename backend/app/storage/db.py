from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS trace_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(task_id) REFERENCES tasks(id)
                );
                """
            )

    def create_task(self, query: str, task_type: str) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO tasks(query, task_type, status) VALUES (?, ?, ?)",
                (query, task_type, "running"),
            )
            return int(cur.lastrowid)

    def finish_task(self, task_id: int, result: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, result_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                ("done", json.dumps(result, ensure_ascii=False), task_id),
            )

    def add_trace(self, task_id: int, stage: str, message: str, data: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO trace_events(task_id, stage, message, data_json)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, stage, message, json.dumps(data, ensure_ascii=False)),
            )

    def get_task(self, task_id: int) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row is None:
                return {}
            item = dict(row)
            item["result"] = json.loads(item.pop("result_json") or "{}")
            return item

    def get_trace(self, task_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trace_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            ).fetchall()
            events = []
            for row in rows:
                item = dict(row)
                item["data"] = json.loads(item.pop("data_json"))
                events.append(item)
            return events
