import json
import tempfile
import unittest
from pathlib import Path

from writing_coach_agent import WritingCoachAgent
from writing_coach_agent.rendering import render_highlighted_essay
from writing_coach_agent.tools import inspect_text, locate_evidence


ROOT = Path(__file__).resolve().parents[1]


class ScriptedJSONBackend:
    """Deterministic test adapter that never participates in production wiring."""

    name = "scripted-test-backend"

    def generate_json(self, system, user, schema):
        task = schema["task"]
        if task == "plan":
            return {"goal": "grounded diagnosis", "steps": [{"tool": "inspect_text", "reason": "inspect draft"}]}
        if task == "report":
            return {
                "summary": "The claim needs more evidence.",
                "scores": {
                    "language": {"score": 3, "rationale": "Mostly clear.", "evidence_sentence_ids": [1]},
                    "argumentation": {"score": 2, "rationale": "Evidence is limited.", "evidence_sentence_ids": [1]},
                },
                "strengths": ["The position is recognizable."],
                "priorities": [{"issue": "Limited evidence", "evidence_sentence_id": 1, "action": "Add an example.", "example": "For example, ..."}],
                "highlights": [{"sentence_id": 1, "label": "needs_evidence", "reason": "Add evidence."}],
                "revision_plan": ["Add an example", "Explain its relevance"],
                "confidence": 0.8,
            }
        if task == "reflect":
            return {"decision": "accept", "reason": "Grounded and actionable.", "repair_instruction": ""}
        raise ValueError(task)


class WritingCoachAgentTests(unittest.TestCase):
    def test_tools_return_grounded_sentence_ids(self):
        essay = "Students should read because books build knowledge. For example, biographies teach history."
        self.assertEqual(inspect_text(essay)["sentence_count"], 2)
        evidence = locate_evidence(essay)["sentence_evidence"]
        self.assertEqual(evidence[0]["labels"], ["claim", "reason"])
        self.assertIn("example", evidence[1]["labels"])

    def test_agent_completes_full_loop_with_test_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            coach = WritingCoachAgent(
                backend=ScriptedJSONBackend(),
                rubric_path=ROOT / "data" / "rubric.jsonl",
                checkpoint_dir=Path(directory),
            )
            run = coach.run("Should students read daily?", "Students should read daily. Books help students learn.")
            self.assertIsNotNone(run.report)
            self.assertIn("load_rubric", run.artifacts)
            self.assertIn("locate_evidence", run.artifacts)
            self.assertTrue((Path(directory) / f"{run.run_id}.json").is_file())
            saved = json.loads((Path(directory) / f"{run.run_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["run_id"], run.run_id)

    def test_renderer_escapes_student_html(self):
        report = {"highlights": [{"sentence_id": 1, "label": "language", "reason": "check"}]}
        rendered = render_highlighted_essay("<script>alert(1)</script>.", report)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)


if __name__ == "__main__":
    unittest.main()
