from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from app.retrieval.hybrid import Document


GITHUB_REPO_RE = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
)
API_ROOT = "https://api.github.com"


@dataclass(frozen=True)
class RepoRef:
    owner: str
    repo: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.full_name}"


class GitHubTool:
    """Read public GitHub repository metadata, README, and directory structure."""

    def __init__(self, timeout_seconds: int = 12) -> None:
        self.timeout_seconds = timeout_seconds

    def load_seed_documents(self, query: str, sources: list[str]) -> list[Document]:
        repo_refs = self._extract_repo_refs(" ".join([query, *sources]))
        if repo_refs:
            docs: list[Document] = []
            for ref in repo_refs:
                try:
                    docs.extend(self.load_repo_documents(ref))
                except requests.RequestException as exc:
                    docs.append(
                        {
                            "source": ref.html_url,
                            "title": f"{ref.full_name} fetch error",
                            "content": f"GitHub fetch failed for {ref.full_name}: {exc}",
                        }
                    )
            if docs:
                return docs
        return self._fallback_documents(query, sources)

    def load_repo_documents(self, ref: RepoRef) -> list[Document]:
        repo = self._get_json(f"/repos/{ref.full_name}")
        default_branch = repo.get("default_branch") or "main"
        docs = [
            self._repo_summary_doc(ref, repo),
            self._tree_doc(ref, default_branch),
        ]
        readme = self._readme_doc(ref, default_branch)
        if readme:
            docs.insert(1, readme)
        return docs

    def _repo_summary_doc(self, ref: RepoRef, repo: dict[str, Any]) -> Document:
        content = "\n".join(
            [
                f"Repository: {repo.get('full_name', ref.full_name)}",
                f"Description: {repo.get('description') or ''}",
                f"Language: {repo.get('language') or ''}",
                f"Default branch: {repo.get('default_branch') or ''}",
                f"Stars: {repo.get('stargazers_count') or 0}",
                f"Forks: {repo.get('forks_count') or 0}",
                f"Open issues: {repo.get('open_issues_count') or 0}",
                f"Topics: {', '.join(repo.get('topics') or [])}",
            ]
        )
        return {
            "source": ref.html_url,
            "title": f"{ref.full_name} repository summary",
            "content": content,
        }

    def _readme_doc(self, ref: RepoRef, default_branch: str) -> Document | None:
        try:
            payload = self._get_json(f"/repos/{ref.full_name}/readme")
        except requests.RequestException:
            return None
        download_url = payload.get("download_url")
        if not download_url:
            return None
        text = self._get_text(download_url)
        return {
            "source": payload.get("html_url") or f"{ref.html_url}/blob/{default_branch}/README.md",
            "title": f"{ref.full_name} README",
            "content": text[:20000],
        }

    def _tree_doc(self, ref: RepoRef, default_branch: str) -> Document:
        payload = self._get_json(f"/repos/{ref.full_name}/git/trees/{default_branch}?recursive=1")
        tree = payload.get("tree") or []
        paths = [item for item in tree if isinstance(item.get("path"), str)]
        selected = self._select_tree_paths(paths)
        content = "\n".join(selected)
        return {
            "source": f"{ref.html_url}/tree/{default_branch}",
            "title": f"{ref.full_name} directory structure",
            "content": content,
        }

    def _select_tree_paths(self, tree: list[dict[str, Any]], limit: int = 120) -> list[str]:
        skip_parts = {
            ".git",
            ".venv",
            "__pycache__",
            "node_modules",
            "dist",
            "build",
            ".next",
            ".cache",
        }
        preferred_ext = {
            ".py",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".go",
            ".java",
            ".md",
            ".toml",
            ".yaml",
            ".yml",
            ".json",
        }
        rows: list[tuple[int, str]] = []
        for item in tree:
            path = item["path"]
            parts = set(path.split("/"))
            if parts & skip_parts:
                continue
            path_lower = path.lower()
            depth = path.count("/")
            score = depth
            if item.get("type") == "tree":
                score -= 2
            if any(path_lower.endswith(ext) for ext in preferred_ext):
                score -= 3
            if path_lower.endswith(("readme.md", "pyproject.toml", "package.json", "requirements.txt")):
                score -= 4
            rows.append((score, path))
        return [path for _score, path in sorted(rows, key=lambda row: (row[0], row[1]))[:limit]]

    def _extract_repo_refs(self, text: str) -> list[RepoRef]:
        seen: set[str] = set()
        refs: list[RepoRef] = []
        for match in GITHUB_REPO_RE.finditer(text):
            repo = match.group("repo").removesuffix(".git")
            ref = RepoRef(match.group("owner"), repo)
            if ref.full_name.lower() not in seen:
                seen.add(ref.full_name.lower())
                refs.append(ref)
        return refs

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "devresearch-agent",
        }
        token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_json(self, path_or_url: str) -> dict[str, Any]:
        url = path_or_url if path_or_url.startswith("http") else f"{API_ROOT}{path_or_url}"
        response = requests.get(url, headers=self._headers(), timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _get_text(self, url: str) -> str:
        response = requests.get(url, headers={"User-Agent": "devresearch-agent"}, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    def _fallback_documents(self, query: str, sources: list[str]) -> list[Document]:
        docs: list[Document] = [
            {
                "source": "internal://agent-architecture",
                "title": "Agent state flow",
                "content": "LangGraph AgentScope ReAct Function Calling Planner Retriever Tool Caller Critic Evaluator state graph replan trace",
            },
            {
                "source": "internal://rag-patterns",
                "title": "RAG retrieval patterns",
                "content": "BM25 Dense Retrieval FAISS Chroma RRF reranker BGE-M3 evidence citation chunk recall MRR nDCG",
            },
            {
                "source": "internal://github-analysis",
                "title": "GitHub repository analysis",
                "content": "README issue pull request code search repository entrypoint module dependency risk test docker run command",
            },
            {
                "source": "internal://agentic-eval",
                "title": "Agentic Eval",
                "content": "tool call accuracy citation support groundedness report completeness factual consistency badcase DeepEval RAGAS",
            },
            {
                "source": "internal://mcp-tools",
                "title": "MCP tool abstraction",
                "content": "MCP server tool schema input output timeout error fallback permission async sync adapter",
            },
        ]
        for source in sources:
            docs.append(
                {
                    "source": f"user://{source}",
                    "title": f"User source {source}",
                    "content": f"{source} related evidence for {query}",
                }
            )
        return docs
