"""Model backend adapters."""

from .base import JSONBackend
from .huggingface import HuggingFaceJSONBackend

__all__ = [
    "HuggingFaceJSONBackend",
    "JSONBackend",
]
