"""Explicit model fallback adapter."""
from __future__ import annotations

from typing import Any, Dict

from .base import JSONBackend


class FallbackJSONBackend:
    """Switch adapters only when the Agent explicitly exhausts its recovery budget."""

    def __init__(self, primary: JSONBackend, fallback: JSONBackend) -> None:
        self.primary = primary
        self.fallback = fallback
        self.degraded = False
        self.fallback_reason: str | None = None

    @property
    def name(self) -> str:
        return self.fallback.name if self.degraded else self.primary.name

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        backend = self.fallback if self.degraded else self.primary
        return backend.generate_json(system, user, schema)

    def activate_fallback(self, reason: str) -> None:
        self.degraded = True
        self.fallback_reason = reason

    def reset(self) -> None:
        self.degraded = False
        self.fallback_reason = None
