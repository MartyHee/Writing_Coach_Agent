"""Deterministic non-AI backend used only after explicit degradation."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from ..tools import split_sentences


class RuleBasedJSONBackend:
    name = "rule-based-fallback (not AI)"

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        task = schema.get("task")
        if task == "plan":
            return {
                "goal": "基于量表和原文证据完成可恢复诊断",
                "steps": [
                    {"tool": "inspect_text", "reason": "提取可验证文本事实"},
                    {"tool": "load_rubric", "reason": "评分必须读取量表"},
                    {"tool": "retrieve_evidence", "reason": "反馈必须引用原文证据"},
                ],
            }
        if task == "report":
            essay = self._essay_from_context(user)
            sentences = split_sentences(essay)
            word_count = len(re.findall(r"\b[A-Za-z']+\b", essay))
            has_example = bool(re.search(r"\b(for example|for instance|such as)\b", essay, re.I))
            has_counter = bool(re.search(r"\b(however|although|some people)\b", essay, re.I))
            language = min(5.0, round(1.5 + word_count / 45, 1))
            argument = min(5.0, round(1.4 + int(has_example) + int(has_counter) + min(word_count / 100, 1), 1))
            evidence_id = 1 if sentences else 0
            return {
                "summary": "立场可识别，但论证链还需要更具体的证据与解释。",
                "scores": {
                    "language": {"score": language, "rationale": "根据清晰度、词汇和句式综合判断。", "evidence_sentence_ids": [evidence_id]},
                    "argumentation": {"score": argument, "rationale": "根据主张、理由、例证和反方回应综合判断。", "evidence_sentence_ids": [evidence_id]},
                },
                "strengths": ["已经表达了可识别的立场。"],
                "priorities": [{
                    "issue": "论证展开不足",
                    "evidence_sentence_id": evidence_id,
                    "action": "补充一个具体例子，并解释它如何支持主张。",
                    "example": "For example, ... This shows that ...",
                }],
                "highlights": [{"sentence_id": evidence_id, "label": "needs_evidence", "reason": "该句需要更多证据。"}],
                "revision_plan": ["补充具体例子", "解释例子与主张的关系", "检查句间衔接"],
                "confidence": 0.55,
            }
        if task == "reflect":
            return {"decision": "accept", "reason": "降级报告字段完整且包含证据编号。", "repair_instruction": ""}
        raise ValueError(f"规则后端不支持任务: {task}")

    @staticmethod
    def _essay_from_context(user: str) -> str:
        try:
            payload = json.loads(user.split("\n上一轮修复要求：", 1)[0])
            return str(payload.get("essay", user))
        except (json.JSONDecodeError, TypeError):
            return user
