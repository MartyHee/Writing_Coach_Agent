"""Convert agent reports into UI-specific views."""
from __future__ import annotations
import json
from typing import Any, Dict


def score_cards(report: Dict[str, Any]) -> str:
    language = report["scores"]["language"]
    argument = report["scores"]["argumentation"]
    badge = (
        "<span class='mode degraded'>Fallback 规则后端</span>"
        if report.get("degraded")
        else "<span class='mode ai'>本地开源模型</span>"
    )
    return f"""
    <div class="score-wrap">
      <div class="score-card"><span>Language</span><strong>{float(language['score']):.1f}</strong><em>/ 5</em><p>{language['rationale']}</p></div>
      <div class="score-card"><span>Argumentation</span><strong>{float(argument['score']):.1f}</strong><em>/ 5</em><p>{argument['rationale']}</p></div>
    </div>
    <div class="run-meta">{badge} Run {report['run_id']} · {report['model_backend']}</div>
    """


def feedback_markdown(report: Dict[str, Any]) -> str:
    lines = [f"### 诊断摘要\n{report['summary']}", "### 优先修订"]
    for index, item in enumerate(report.get("priorities", []), 1):
        lines.extend([
            f"**{index}. {item['issue']}** · 句 {item['evidence_sentence_id']}",
            f"- 动作：{item['action']}",
            f"- 局部示例：`{item['example']}`",
        ])
    lines.append("### 下一轮 Plan")
    lines.extend(f"{index}. {step}" for index, step in enumerate(report.get("revision_plan", []), 1))
    return "\n\n".join(lines)


def trace_rows(trace: list[Dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            index + 1,
            event.get("time", ""),
            event.get("event", ""),
            json.dumps({key: value for key, value in event.items() if key not in {"time", "event"}}, ensure_ascii=False),
        )
        for index, event in enumerate(trace)
    ]
