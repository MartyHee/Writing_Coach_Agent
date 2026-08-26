"""TF-IDF and MiniLM evidence retrieval with rank fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence


@dataclass(frozen=True)
class RankedCandidate:
    index: int
    score: float
    backend_scores: Dict[str, float]


class Retriever(Protocol):
    name: str

    def rank(self, query: str, candidates: Sequence[str]) -> List[RankedCandidate]: ...


class TfidfRetriever:
    name = "tfidf"

    def rank(self, query: str, candidates: Sequence[str]) -> List[RankedCandidate]:
        if not candidates:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform([query, *candidates])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        order = sorted(range(len(candidates)), key=lambda index: float(scores[index]), reverse=True)
        return [RankedCandidate(index, float(scores[index]), {self.name: float(scores[index])}) for index in order]


class MiniLMRetriever:
    name = "all-MiniLM-L6-v2"

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def rank(self, query: str, candidates: Sequence[str]) -> List[RankedCandidate]:
        if not candidates:
            return []
        vectors = self._load().encode([query, *candidates], normalize_embeddings=True)
        scores = vectors[1:] @ vectors[0]
        order = sorted(range(len(candidates)), key=lambda index: float(scores[index]), reverse=True)
        return [RankedCandidate(index, float(scores[index]), {self.name: float(scores[index])}) for index in order]


class DualRetriever:
    """Run both retrievers and combine independent rankings with reciprocal-rank fusion."""

    name = "tfidf+all-MiniLM-L6-v2"

    def __init__(self, lexical: Retriever | None = None, semantic: Retriever | None = None, rrf_k: int = 60) -> None:
        self.lexical = lexical or TfidfRetriever()
        self.semantic = semantic or MiniLMRetriever()
        self.rrf_k = rrf_k

    def rank(self, query: str, candidates: Sequence[str]) -> List[RankedCandidate]:
        if not candidates:
            return []
        rankings = [self.lexical.rank(query, candidates), self.semantic.rank(query, candidates)]
        fused = {index: 0.0 for index in range(len(candidates))}
        details: Dict[int, Dict[str, float]] = {index: {} for index in range(len(candidates))}
        for ranking in rankings:
            for rank, candidate in enumerate(ranking, 1):
                fused[candidate.index] += 1.0 / (self.rrf_k + rank)
                details[candidate.index].update(candidate.backend_scores)
        order = sorted(fused, key=fused.get, reverse=True)
        return [RankedCandidate(index, fused[index], details[index]) for index in order]
