"""Safe HTML rendering for evidence highlights."""
import html
from typing import Any, Dict

from .tools import split_sentences

LABEL_STYLE = {
    "strength": ("#dcfce7", "#166534", "亮点"),
    "needs_evidence": ("#fef3c7", "#92400e", "需补证据"),
    "language": ("#fee2e2", "#991b1b", "语言"),
    "counterargument": ("#e0e7ff", "#3730a3", "反方回应"),
}


def render_highlighted_essay(essay: str, report: Dict[str, Any]) -> str:
    by_id = {int(item["sentence_id"]): item for item in report.get("highlights", [])}
    pieces = ['<div class="essay-paper">']
    for sentence_id, sentence in enumerate(split_sentences(essay), 1):
        mark = by_id.get(sentence_id)
        if mark:
            background, color, label = LABEL_STYLE.get(mark.get("label"), ("#f1f5f9", "#334155", "关注"))
            title = html.escape(str(mark.get("reason", "")), quote=True)
            pieces.append(
                f'<span class="sentence-mark" title="{title}" style="background:{background};color:{color}">'
                f"{html.escape(sentence)}<small>{label}</small></span> "
            )
        else:
            pieces.append(f"<span>{html.escape(sentence)}</span> ")
    pieces.append("</div>")
    return "".join(pieces)
