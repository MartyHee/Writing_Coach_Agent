"""Runtime domain models."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentRun:
    prompt: str
    essay: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    plan: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None

    def log(self, event: str, **details: Any) -> None:
        self.trace.append({"time": time.strftime("%H:%M:%S"), "event": event, **details})
