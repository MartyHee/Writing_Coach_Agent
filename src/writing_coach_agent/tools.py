"""Deterministic tools used by the agent executor."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict

Tool = Callable[[str, Path], Dict[str, Any]]


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
    return {"sentence_evidence": rows}


def default_tools() -> Dict[str, Tool]:
    return {
        "inspect_text": lambda essay, _: inspect_text(essay),
        "load_rubric": lambda _, rubric_path: load_rubric(rubric_path),
        "locate_evidence": lambda essay, _: locate_evidence(essay),
    }


TOOLS = default_tools()
