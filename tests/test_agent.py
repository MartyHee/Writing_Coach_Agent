import json
import tempfile
import unittest
from pathlib import Path

from writing_coach_agent import FallbackJSONBackend, RuleBasedJSONBackend, WritingCoachAgent
from writing_coach_agent.checkpoints import CheckpointStore
from writing_coach_agent.retrieval import DualRetriever, RankedCandidate
from writing_coach_agent.rendering import render_highlighted_essay
from writing_coach_agent.tools import default_tools, inspect_text, locate_evidence


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


class StaticRetriever:
    def __init__(self, name, order):
        self.name = name
        self.order = order

    def rank(self, query, candidates):
        return [RankedCandidate(index, 1.0 / rank, {self.name: 1.0 / rank}) for rank, index in enumerate(self.order, 1)]


class ReplanningBackend(ScriptedJSONBackend):
    def __init__(self):
        self.planning_calls = 0

    def generate_json(self, system, user, schema):
        if schema["task"] == "plan":
            self.planning_calls += 1
            return {"goal": "recover", "steps": [{"tool": "unstable_tool", "reason": "try evidence tool"}]}
        return super().generate_json(system, user, schema)


class FailingBackend:
    name = "failing-primary"

    def generate_json(self, system, user, schema):
        raise ConnectionError("model service 503")


class InvalidReportBackend(ScriptedJSONBackend):
    name = "invalid-schema-primary"

    def generate_json(self, system, user, schema):
        if schema["task"] == "report":
            return {"summary": "missing scores"}
        return super().generate_json(system, user, schema)


class InvalidPlanBackend(ScriptedJSONBackend):
    name = "invalid-plan-primary"

    def generate_json(self, system, user, schema):
        if schema["task"] == "plan":
            return {"goal": "missing steps"}
        return super().generate_json(system, user, schema)


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
            self.assertIn("tool_results", saved["memory"]["working"])
            self.assertTrue(saved["memory"]["episodic"])
            restored = CheckpointStore(Path(directory)).load(run.run_id)
            self.assertEqual(restored.memory.working, run.memory.working)

    def test_dual_retriever_fuses_both_rankings(self):
        retriever = DualRetriever(
            lexical=StaticRetriever("tfidf", [0, 1, 2]),
            semantic=StaticRetriever("minilm", [2, 1, 0]),
            rrf_k=10,
        )
        result = retriever.rank("query", ["a", "b", "c"])
        self.assertEqual({item.index for item in result}, {0, 1, 2})
        self.assertEqual(set(result[0].backend_scores), {"tfidf", "minilm"})

    def test_tool_failure_triggers_memory_aware_replan(self):
        backend = ReplanningBackend()

        def unstable_tool(essay, rubric_path):
            raise ValueError("evidence index unavailable")

        tools = default_tools()
        tools["unstable_tool"] = unstable_tool
        coach = WritingCoachAgent(
            backend=backend,
            rubric_path=ROOT / "data" / "rubric.jsonl",
            tools=tools,
            max_replans=1,
        )
        run = coach.run("Should students read?", "Students should read. Books build knowledge.")
        self.assertEqual(backend.planning_calls, 2)
        self.assertTrue(any(event["event"] == "replan_created" for event in run.trace))
        self.assertEqual(run.memory.working["tool_failures"][0]["tool"], "unstable_tool")

    def test_model_failures_activate_explicit_fallback(self):
        backend = FallbackJSONBackend(FailingBackend(), RuleBasedJSONBackend())
        coach = WritingCoachAgent(
            backend=backend,
            rubric_path=ROOT / "data" / "rubric.jsonl",
            max_retries=1,
        )
        run = coach.run("Should students read?", "Students should read. Books build knowledge.")
        self.assertTrue(run.report["degraded"])
        self.assertIn("ConnectionError", run.report["fallback_reason"])
        self.assertEqual(run.report["model_backend"], "rule-based-fallback (not AI)")
        self.assertTrue(any(event["event"] == "fallback_activated" for event in run.trace))

    def test_invalid_report_schema_activates_fallback(self):
        backend = FallbackJSONBackend(InvalidReportBackend(), ScriptedJSONBackend())
        coach = WritingCoachAgent(
            backend=backend,
            rubric_path=ROOT / "data" / "rubric.jsonl",
            max_repairs=0,
        )
        run = coach.run("Should students read?", "Students should read. Books build knowledge.")
        self.assertTrue(run.report["degraded"])
        self.assertIn("missing", run.report["fallback_reason"])
        self.assertTrue(any(event["event"] == "schema_fallback_succeeded" for event in run.trace))

    def test_invalid_plan_schema_activates_fallback(self):
        backend = FallbackJSONBackend(InvalidPlanBackend(), ScriptedJSONBackend())
        coach = WritingCoachAgent(backend=backend, rubric_path=ROOT / "data" / "rubric.jsonl")
        run = coach.run("Should students read?", "Students should read. Books build knowledge.")
        self.assertTrue(run.report["degraded"])
        self.assertTrue(any(event["event"] == "plan_schema_fallback_succeeded" for event in run.trace))

    def test_renderer_escapes_student_html(self):
        report = {"highlights": [{"sentence_id": 1, "label": "language", "reason": "check"}]}
        rendered = render_highlighted_essay("<script>alert(1)</script>.", report)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)


if __name__ == "__main__":
    unittest.main()
