"""Public interface for the Writing Coach Agent package."""

from .agent import WritingCoachAgent
from .backends import HuggingFaceJSONBackend, JSONBackend
from .models import AgentRun
from .rendering import render_highlighted_essay

__all__ = [
    "AgentRun",
    "HuggingFaceJSONBackend",
    "JSONBackend",
    "WritingCoachAgent",
    "render_highlighted_essay",
]
