"""Stable structured-output contracts shared by planner and executor."""

PLAN_SCHEMA = {
    "task": "plan",
    "goal": "string",
    "steps": [{"tool": "one allowed tool", "reason": "string"}],
}

REPORT_SCHEMA = {
    "task": "report",
    "summary": "string",
    "scores": {
        "language": {"score": "1..5", "rationale": "string", "evidence_sentence_ids": [1]},
        "argumentation": {"score": "1..5", "rationale": "string", "evidence_sentence_ids": [1]},
    },
    "strengths": ["string"],
    "priorities": [{
        "issue": "string",
        "evidence_sentence_id": 1,
        "action": "string",
        "example": "short fragment, do not rewrite essay",
    }],
    "highlights": [{
        "sentence_id": 1,
        "label": "strength|needs_evidence|language|counterargument",
        "reason": "string",
    }],
    "revision_plan": ["ordered action"],
    "confidence": "0..1",
}

REFLECT_SCHEMA = {
    "task": "reflect",
    "decision": "accept|revise",
    "reason": "string",
    "repair_instruction": "string",
}
