"""FastAPI application factory and dependency wiring."""
from typing import Any, Dict, Optional

from ..agent import WritingCoachAgent
from ..backends import HuggingFaceJSONBackend
from ..config import Settings


def build_agent(settings: Settings) -> WritingCoachAgent:
    return WritingCoachAgent(
        backend=HuggingFaceJSONBackend(settings.model_id),
        rubric_path=settings.rubric_path,
        checkpoint_dir=settings.checkpoint_dir,
    )


def create_app(settings: Optional[Settings] = None, coach: Optional[WritingCoachAgent] = None):
    from fastapi import FastAPI
    from pydantic import BaseModel

    settings = settings or Settings.from_env()
    coach = coach or build_agent(settings)
    api = FastAPI(title="Writing Coach Agent", version="3.0.0")

    class DiagnoseRequest(BaseModel):
        prompt: str
        essay: str

    @api.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "backend": coach.backend.name}

    @api.post("/api/diagnose")
    def diagnose(request: DiagnoseRequest) -> Dict[str, Any]:
        return coach.run(request.prompt, request.essay).report

    try:
        import gradio as gr
        from .ui import build_demo
        return gr.mount_gradio_app(api, build_demo(coach), path="/")
    except ImportError:
        return api
