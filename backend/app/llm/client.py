from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is in requirements
    load_dotenv = None


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 45


class LLMClient:
    """Small OpenAI-compatible chat completions client."""

    def __init__(self) -> None:
        if load_dotenv:
            load_dotenv()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_ID") or "gpt-4o-mini"
        self.config = LLMConfig(api_key=api_key, base_url=base_url, model=model)

    @property
    def enabled(self) -> bool:
        return bool(self.config.api_key)

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("LLM_API_KEY is not configured")
        return await asyncio.to_thread(self._chat_sync, system, user, temperature, max_tokens)

    async def json_chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> dict[str, Any]:
        text = await self.chat(system, user, temperature=temperature, max_tokens=max_tokens)
        return parse_json_object(text)

    def _chat_sync(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return a JSON object: {text[:200]}")
    return json.loads(cleaned[start : end + 1])
