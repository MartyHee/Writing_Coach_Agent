"""Run persistence adapter."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import AgentRun


class CheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, run: AgentRun) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{run.run_id}.json"
        target.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2), encoding="utf-8")
        return target
