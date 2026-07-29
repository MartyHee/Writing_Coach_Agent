"""Writing Coach Studio: FastAPI service + Gradio product interface.

Run from the notebooks directory:
    python app.py

The service listens on 0.0.0.0:7860 so trusted devices on the same LAN can
open http://<teacher-lan-ip>:7860. The FastAPI routes and Gradio page share
the same process and the same model-backed Agent instance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_core import DemoJSONBackend, WritingCoachAgent, create_backend, render_highlighted_essay

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
USE_LOCAL_MODEL = os.getenv("WRITING_COACH_USE_LOCAL_MODEL", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
backend = create_backend(allow_offline_fallback=True) if USE_LOCAL_MODEL else DemoJSONBackend()
coach = WritingCoachAgent(
    backend=backend,
    rubric_path=PROJECT_ROOT / "data" / "rubric.json",
    checkpoint_dir=PROJECT_ROOT / "outputs" / "product_runs",
)


def _score_cards(report: dict[str, Any]) -> str:
    language = report["scores"]["language"]
    argument = report["scores"]["argumentation"]
    degraded = report.get("degraded", False)
    badge = "<span class='mode degraded'>离线演示模式</span>" if degraded else "<span class='mode ai'>本地开源模型</span>"
    return f"""
    <div class="score-wrap">
      <div class="score-card"><span>Language</span><strong>{float(language['score']):.1f}</strong><em>/ 5</em><p>{language['rationale']}</p></div>
      <div class="score-card"><span>Argumentation</span><strong>{float(argument['score']):.1f}</strong><em>/ 5</em><p>{argument['rationale']}</p></div>
    </div>
    <div class="run-meta">{badge} Run {report['run_id']} · {report['model_backend']}</div>
    """


def _feedback_markdown(report: dict[str, Any]) -> str:
    lines = [f"### 诊断摘要\n{report['summary']}", "### 优先修订"]
    for i, item in enumerate(report.get("priorities", []), 1):
        lines.extend([
            f"**{i}. {item['issue']}** · 句 {item['evidence_sentence_id']}",
            f"- 动作：{item['action']}",
            f"- 局部示例：`{item['example']}`",
        ])
    lines.append("### 下一轮 Plan")
    lines.extend(f"{i}. {step}" for i, step in enumerate(report.get("revision_plan", []), 1))
    return "\n\n".join(lines)


def diagnose_for_ui(prompt: str, essay: str):
    run = coach.run(prompt.strip(), essay.strip())
    report = run.report
    # Gradio Dataframe expects rows (tuples/lists), not a list of dictionaries.
    # Keep the event-specific fields in one JSON column so no trace details are
    # lost when different events have different keys.
    trace_rows = [
        (
            i + 1,
            event.get("time", ""),
            event.get("event", ""),
            json.dumps(
                {k: v for k, v in event.items() if k not in {"time", "event"}},
                ensure_ascii=False,
            ),
        )
        for i, event in enumerate(run.trace)
    ]
    return (
        _score_cards(report),
        render_highlighted_essay(essay, report),
        _feedback_markdown(report),
        trace_rows,
        json.dumps(report, ensure_ascii=False, indent=2),
    )


CSS = """
:root { --ink:#172033; --muted:#64748b; --brand:#6750e8; }
.gradio-container { max-width: 1220px !important; background:linear-gradient(150deg,#f8f7ff,#f8fafc 45%,#eff6ff); }
.hero { padding:24px 26px; border:1px solid #e7e5ff; border-radius:22px; background:rgba(255,255,255,.85); box-shadow:0 16px 45px rgba(66,55,140,.08); }
.hero h1 { color:var(--ink); margin:0 0 6px; font-size:30px; } .hero p { color:var(--muted); margin:0; }
.panel { border-radius:18px !important; border:1px solid #e8eaf2 !important; background:white !important; }
.score-wrap { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.score-card { padding:18px; border:1px solid #e8e7f8; border-radius:16px; background:linear-gradient(145deg,#fff,#faf9ff); }
.score-card span { display:block;color:#64748b;font-weight:650; }.score-card strong {font-size:38px;color:#332879}.score-card em{color:#94a3b8;font-style:normal}.score-card p{font-size:13px;color:#526074;min-height:38px}
.run-meta { margin:10px 2px; color:#64748b;font-size:12px }.mode{padding:4px 8px;border-radius:999px;margin-right:7px}.mode.ai{background:#dcfce7;color:#166534}.mode.degraded{background:#fef3c7;color:#92400e}
.essay-paper { font-family:Georgia,serif;font-size:17px;line-height:2.2;color:#111827 !important;background:#ffffff;padding:24px;border-radius:16px;border:1px solid #e7eaf0;min-height:170px }
.essay-paper > span:not(.sentence-mark) { color:#111827 !important; }
.feedback-panel { color:#111827 !important; background:#ffffff !important; }
.feedback-panel * { color:#111827 !important; }
.feedback-panel code { background:#f3f4f6 !important; color:#111827 !important; }
.sentence-mark { padding:3px 5px;border-radius:6px;box-decoration-break:clone;-webkit-box-decoration-break:clone;cursor:help }.sentence-mark small{font-family:system-ui;font-size:10px;margin-left:5px;padding:2px 5px;border:1px solid currentColor;border-radius:8px;vertical-align:middle}
#analyze-btn { background:linear-gradient(135deg,#6750e8,#496fe5);border:0; }
"""


def build_demo():
    import gradio as gr

    with gr.Blocks(css=CSS, title="Writing Coach Studio") as demo:
        gr.HTML("<div class='hero'><h1>✍️ Writing Coach Studio</h1><p>Agent 自主规划 · Rubric 评分 · 原文证据 · 可执行修订</p></div>")
        with gr.Row():
            with gr.Column(scale=5, elem_classes="panel"):
                prompt = gr.Textbox(label="Writing prompt", value="Should students receive cash rewards for good grades?")
                essay = gr.Textbox(label="Student draft", lines=12, placeholder="Paste an argumentative essay here...")
                analyze = gr.Button("运行 Agent 诊断", variant="primary", elem_id="analyze-btn")
            with gr.Column(scale=4, elem_classes="panel"):
                score_cards = gr.HTML("<p style='padding:20px;color:#64748b'>运行后显示评分与模型状态。</p>")
                feedback = gr.Markdown("诊断建议将显示在这里。", elem_classes="feedback-panel")
        with gr.Tabs():
            with gr.Tab("原文证据高亮"):
                highlighted = gr.HTML("<div class='essay-paper'>等待诊断…</div>")
            with gr.Tab("Agent Trace"):
                trace = gr.Dataframe(
                    headers=["#", "time", "event", "details"],
                    datatype=["number", "str", "str", "str"],
                    label="可观测执行轨迹",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("完整 JSON"):
                raw = gr.Code(label="Structured report", language="json")
        analyze.click(diagnose_for_ui, [prompt, essay], [score_cards, highlighted, feedback, trace, raw])
        gr.Examples(
            examples=[["Should students receive cash rewards for good grades?", "Students should get money for grades. Money is useful. Good grades are good. Parents can give money and students will happy. This is my opinion."]],
            inputs=[prompt, essay],
        )
    return demo


try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    api = FastAPI(title="Writing Coach Agent", version="2.0.0")

    class DiagnoseRequest(BaseModel):
        prompt: str
        essay: str

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": backend.name}

    @api.post("/api/diagnose")
    def diagnose(request: DiagnoseRequest) -> dict[str, Any]:
        return coach.run(request.prompt, request.essay).report

    try:
        import gradio as gr
        app = gr.mount_gradio_app(api, build_demo(), path="/")
    except ImportError:
        app = api
except ImportError:
    app = None


if __name__ == "__main__":
    import uvicorn

    # Use the mounted FastAPI + Gradio application, rather than launching a
    # second Gradio-only server. This keeps /health, /api/diagnose and the UI
    # available on one LAN port.
    # If another classroom service already owns 7860, start with
    # $env:WRITING_COACH_PORT="7861"; python app.py
    port = int(os.getenv("WRITING_COACH_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
