"""Run-scoped working and episodic memory."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentMemory:
    """Keep current artifacts and an append-only decision history for one run."""

    working: Dict[str, Any] = field(default_factory=dict)
    episodic: List[Dict[str, Any]] = field(default_factory=list)

    def remember(self, event: str, **payload: Any) -> None:
        self.episodic.append({"time": time.strftime("%H:%M:%S"), "event": event, **payload})

    def remember_plan(self, plan: List[Dict[str, Any]], reason: str = "initial") -> None:
        self.working["plan"] = plan
        self.remember("plan_remembered", reason=reason, plan=plan)

    def remember_tool_result(self, tool: str, result: Any) -> None:
        self.working.setdefault("tool_results", {})[tool] = result
        self.remember("tool_result_remembered", tool=tool)

    def remember_failure(self, tool: str, error: str, decision: str) -> None:
        self.working.setdefault("tool_failures", []).append({"tool": tool, "error": error})
        self.remember("tool_failure_remembered", tool=tool, error=error, decision=decision)

    def context(self) -> Dict[str, Any]:
        return {"working": self.working, "recent_episodes": self.episodic[-12:]}
