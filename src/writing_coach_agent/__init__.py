"""Public interface for the Writing Coach Agent package."""

from .agent import WritingCoachAgent
from .backends import FallbackJSONBackend, HuggingFaceJSONBackend, JSONBackend, RuleBasedJSONBackend
from .memory import AgentMemory
from .models import AgentRun
from .retrieval import DualRetriever, MiniLMRetriever, TfidfRetriever
from .rendering import render_highlighted_essay

__all__ = [
    "AgentRun",
    "AgentMemory",
    "DualRetriever",
    "FallbackJSONBackend",
    "HuggingFaceJSONBackend",
    "JSONBackend",
    "MiniLMRetriever",
    "RuleBasedJSONBackend",
    "TfidfRetriever",
    "WritingCoachAgent",
    "render_highlighted_essay",
]
