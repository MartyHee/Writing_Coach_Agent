"""Public interface for the Writing Coach Agent package."""

from .agent import WritingCoachAgent
from .backends import HuggingFaceJSONBackend, JSONBackend
from .memory import AgentMemory
from .models import AgentRun
from .retrieval import DualRetriever, MiniLMRetriever, TfidfRetriever
from .rendering import render_highlighted_essay

__all__ = [
    "AgentRun",
    "AgentMemory",
    "DualRetriever",
    "HuggingFaceJSONBackend",
    "JSONBackend",
    "MiniLMRetriever",
    "TfidfRetriever",
    "WritingCoachAgent",
    "render_highlighted_essay",
]
