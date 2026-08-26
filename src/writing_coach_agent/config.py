"""Environment-backed application configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent


@dataclass(frozen=True)
class Settings:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    rubric_path: Path = PROJECT_ROOT / "data" / "rubric.jsonl"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "product_runs"
    host: str = "0.0.0.0"
    port: int = 7860

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_id=os.getenv("WRITING_COACH_MODEL", cls.model_id),
            rubric_path=Path(os.getenv("WRITING_COACH_RUBRIC_PATH", str(cls.rubric_path))),
            checkpoint_dir=Path(os.getenv("WRITING_COACH_CHECKPOINT_DIR", str(cls.checkpoint_dir))),
            host=os.getenv("WRITING_COACH_HOST", cls.host),
            port=int(os.getenv("WRITING_COACH_PORT", str(cls.port))),
        )
