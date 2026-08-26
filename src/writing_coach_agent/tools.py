"""Deterministic tools and the tool-routing registry."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .retrieval import DualRetriever, Retriever

Tool = Callable[[str, Path], Dict[str, Any]]


class ToolRegistry:
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self._tools = dict(tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def call(self, name: str, essay: str, rubric_path: Path) -> Dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"未知工具: {name}")
        result = self._tools[name](essay, rubric_path)
        json.dumps(result, ensure_ascii=False)
        return result


def split_sentences(essay: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", essay.strip()) if item.strip()]


def inspect_text(essay: str) -> Dict[str, Any]:
    sentences = split_sentences(essay)
    words = re.findall(r"\b[A-Za-z']+\b", essay)
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "sentences": [{"id": index + 1, "text": sentence} for index, sentence in enumerate(sentences)],
    }


def load_rubric(rubric_path: Path) -> Dict[str, Any]:
    return json.loads(rubric_path.read_text(encoding="utf-8"))


def locate_evidence(essay: str) -> Dict[str, Any]:
    patterns = {
        "claim": r"\b(should|must|believe|opinion)\b",
        "reason": r"\b(because|since|reason)\b",
        "example": r"\b(for example|for instance|such as)\b",
        "counterargument": r"\b(however|although|some people|on the other hand)\b",
        "conclusion": r"\b(therefore|in conclusion|to conclude)\b",
    }
    rows = []
    for sentence_id, sentence in enumerate(split_sentences(essay), 1):
        labels = [label for label, pattern in patterns.items() if re.search(pattern, sentence, re.I)]
        rows.append({"sentence_id": sentence_id, "labels": labels, "text": sentence})
    return {"sentence_evidence": rows, "backend": "rule-labels"}


def retrieve_evidence(essay: str, retriever: Retriever, top_k: int = 3) -> Dict[str, Any]:
    sentences = split_sentences(essay)
    focus = "main claim reasons concrete evidence counterargument conclusion clear language"
    ranked = retriever.rank(focus, sentences)[:top_k]
    rows = [
        {
            "sentence_id": item.index + 1,
            "text": sentences[item.index],
            "relevance": round(float(item.score), 6),
            "backend_scores": {key: round(float(value), 6) for key, value in item.backend_scores.items()},
        }
        for item in ranked
    ]
    return {"sentence_evidence": rows, "backend": retriever.name}


def default_tools(retriever: Optional[Retriever] = None) -> Dict[str, Tool]:
    tools: Dict[str, Tool] = {
        "inspect_text": lambda essay, _: inspect_text(essay),
        "load_rubric": lambda _, rubric_path: load_rubric(rubric_path),
        "locate_evidence": lambda essay, _: locate_evidence(essay),
    }
    if retriever is not None:
        tools["retrieve_evidence"] = lambda essay, _: retrieve_evidence(essay, retriever)
    return tools


TOOLS = default_tools()


def production_tools() -> Dict[str, Tool]:
    return default_tools(DualRetriever())
