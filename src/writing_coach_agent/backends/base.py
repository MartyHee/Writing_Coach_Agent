"""Backend interface."""
from typing import Any, Dict, runtime_checkable

try:
    from typing import Protocol
except ImportError:  # Python 3.7 compatibility for legacy notebook environments
    from typing_extensions import Protocol


class JSONBackend(Protocol):
    name: str

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        ...


@runtime_checkable
class FallbackCapableJSONBackend(JSONBackend, Protocol):
    degraded: bool
    fallback_reason: str | None

    def activate_fallback(self, reason: str) -> None: ...

    def reset(self) -> None: ...
