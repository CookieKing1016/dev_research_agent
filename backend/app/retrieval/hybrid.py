from __future__ import annotations

import math
import re
from collections import Counter
from typing import TypedDict


class Document(TypedDict):
    source: str
    title: str
    content: str


class ScoredDocument(Document):
    score: float


TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class HybridRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_tokens = [tokenize(doc["content"] + " " + doc["title"]) for doc in documents]
        self.df = Counter(token for tokens in self.doc_tokens for token in set(tokens))

    def search(self, query: str, top_k: int = 5) -> list[ScoredDocument]:
        if not self.documents:
            return []
        bm25 = self._rank_bm25(query)
        dense = self._rank_dense(query)
        fused = self._rrf([bm25, dense])
        results: list[ScoredDocument] = []
        for idx, score in fused[:top_k]:
            doc = self.documents[idx]
            results.append({**doc, "score": round(score, 4)})
        return results

    def _rank_bm25(self, query: str) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        scores = []
        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            score = 0.0
            for token in q_tokens:
                if token not in tf:
                    continue
                idf = math.log((len(self.documents) - self.df[token] + 0.5) / (self.df[token] + 0.5) + 1)
                denom = tf[token] + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / avg_len)
                score += idf * tf[token] * 2.5 / denom
            scores.append((idx, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _rank_dense(self, query: str) -> list[tuple[int, float]]:
        q = Counter(tokenize(query))
        scores = []
        for idx, tokens in enumerate(self.doc_tokens):
            d = Counter(tokens)
            shared = set(q) & set(d)
            numerator = sum(q[token] * d[token] for token in shared)
            q_norm = math.sqrt(sum(value * value for value in q.values())) or 1.0
            d_norm = math.sqrt(sum(value * value for value in d.values())) or 1.0
            scores.append((idx, numerator / (q_norm * d_norm)))
        return sorted(scores, key=lambda item: item[1], reverse=True)

    def _rrf(self, rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
        scores: Counter[int] = Counter()
        for ranking in rankings:
            for rank, (idx, _score) in enumerate(ranking, start=1):
                scores[idx] += 1 / (k + rank)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
