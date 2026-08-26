"""Gradio interface factory."""
import json

from ..agent import WritingCoachAgent
from ..rendering import render_highlighted_essay
from .presenters import feedback_markdown, score_cards, trace_rows

CSS = """
:root { --ink:#172033; --muted:#64748b; --brand:#6750e8; }
.gradio-container { max-width:1220px !important; background:linear-gradient(150deg,#f8f7ff,#f8fafc 45%,#eff6ff); }
.hero { padding:24px 26px; border:1px solid #e7e5ff; border-radius:22px; background:rgba(255,255,255,.85); box-shadow:0 16px 45px rgba(66,55,140,.08); }
.hero h1 { color:var(--ink); margin:0 0 6px; font-size:30px; } .hero p { color:var(--muted); margin:0; }
.panel { border-radius:18px !important; border:1px solid #e8eaf2 !important; background:white !important; }
.score-wrap { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.score-card { padding:18px; border:1px solid #e8e7f8; border-radius:16px; background:linear-gradient(145deg,#fff,#faf9ff); }
.score-card span { display:block;color:#64748b;font-weight:650; }.score-card strong {font-size:38px;color:#332879}.score-card em{color:#94a3b8;font-style:normal}.score-card p{font-size:13px;color:#526074;min-height:38px}
.run-meta { margin:10px 2px; color:#64748b;font-size:12px }.mode{padding:4px 8px;border-radius:999px;margin-right:7px}.mode.ai{background:#dcfce7;color:#166534}
.essay-paper { font-family:Georgia,serif;font-size:17px;line-height:2.2;color:#111827 !important;background:#fff;padding:24px;border-radius:16px;border:1px solid #e7eaf0;min-height:170px }
.essay-paper > span:not(.sentence-mark),.feedback-panel,.feedback-panel * { color:#111827 !important; }
.feedback-panel code { background:#f3f4f6 !important; color:#111827 !important; }
.sentence-mark { padding:3px 5px;border-radius:6px;box-decoration-break:clone;-webkit-box-decoration-break:clone;cursor:help }.sentence-mark small{font-family:system-ui;font-size:10px;margin-left:5px;padding:2px 5px;border:1px solid currentColor;border-radius:8px;vertical-align:middle}
#analyze-btn { background:linear-gradient(135deg,#6750e8,#496fe5);border:0; }
"""


def build_demo(coach: WritingCoachAgent):
    import gradio as gr

    def diagnose_for_ui(prompt: str, essay: str):
        run = coach.run(prompt.strip(), essay.strip())
        report = run.report
        return score_cards(report), render_highlighted_essay(essay, report), feedback_markdown(report), trace_rows(run.trace), json.dumps(report, ensure_ascii=False, indent=2)

    with gr.Blocks(css=CSS, title="Writing Coach Studio") as demo:
        gr.HTML("<div class='hero'><h1>✍️ Writing Coach Studio</h1><p>Agent 自主规划 · Rubric 评分 · 原文证据 · 可执行修订</p></div>")
        with gr.Row():
            with gr.Column(scale=5, elem_classes="panel"):
                prompt = gr.Textbox(label="Writing prompt", value="Should students receive cash rewards for good grades?")
                essay = gr.Textbox(label="Student draft", lines=12, placeholder="Paste an argumentative essay here...")
                analyze = gr.Button("运行 Agent 诊断", variant="primary", elem_id="analyze-btn")
            with gr.Column(scale=4, elem_classes="panel"):
                scores = gr.HTML("<p style='padding:20px;color:#64748b'>运行后显示评分与模型状态。</p>")
                feedback = gr.Markdown("诊断建议将显示在这里。", elem_classes="feedback-panel")
        with gr.Tabs():
            with gr.Tab("原文证据高亮"):
                highlighted = gr.HTML("<div class='essay-paper'>等待诊断…</div>")
            with gr.Tab("Agent Trace"):
                trace = gr.Dataframe(headers=["#", "time", "event", "details"], datatype=["number", "str", "str", "str"], label="可观测执行轨迹", interactive=False, wrap=True)
            with gr.Tab("完整 JSON"):
                raw = gr.Code(label="Structured report", language="json")
        analyze.click(diagnose_for_ui, [prompt, essay], [scores, highlighted, feedback, trace, raw])
        gr.Examples(examples=[["Should students receive cash rewards for good grades?", "Students should get money for grades. Money is useful. Good grades are good. Parents can give money and students will happy. This is my opinion."]], inputs=[prompt, essay])
    return demo
