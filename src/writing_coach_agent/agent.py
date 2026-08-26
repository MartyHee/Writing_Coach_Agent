"""Planner-executor-reflector orchestration."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends.base import JSONBackend
from .checkpoints import CheckpointStore
from .contracts import PLAN_SCHEMA, REFLECT_SCHEMA, REPORT_SCHEMA
from .models import AgentRun
from .prompts import PLANNER_SYSTEM, REFLECTOR_SYSTEM, REPORT_SYSTEM, planner_input
from .tools import TOOLS, Tool, split_sentences


class WritingCoachAgent:
    """Deep module exposing one operation: produce a grounded coaching run."""

    def __init__(
        self,
        backend: JSONBackend,
        rubric_path: Path,
        checkpoint_dir: Optional[Path] = None,
        max_repairs: int = 1,
        max_retries: int = 2,
        tools: Optional[Dict[str, Tool]] = None,
    ) -> None:
        self.backend = backend
        self.rubric_path = rubric_path
        self.checkpoints = CheckpointStore(checkpoint_dir) if checkpoint_dir else None
        self.max_repairs = max_repairs
        self.max_retries = max_retries
        self.tools = tools or TOOLS

    def _save(self, run: AgentRun) -> None:
        if self.checkpoints:
            self.checkpoints.save(run)
            run.log("checkpoint_saved")

    def _valid_plan(self, plan: Dict[str, Any]) -> List[Dict[str, str]]:
        valid = []
        seen = set()
        for step in plan.get("steps", []):
            tool = step.get("tool")
            if tool in self.tools and tool not in seen:
                valid.append({"tool": tool, "reason": str(step.get("reason", ""))})
                seen.add(tool)
        if not valid:
            raise ValueError("Planner 没有选择任何合法工具")
        for required, reason in [
            ("load_rubric", "Executor guardrail: scoring requires the rubric"),
            ("locate_evidence", "Executor guardrail: feedback requires valid sentence evidence"),
        ]:
            if required not in seen:
                if required not in self.tools:
                    raise ValueError(f"Executor 缺少必需工具: {required}")
                valid.append({"tool": required, "reason": reason})
                seen.add(required)
        return valid

    def _ask(
        self,
        run: AgentRun,
        stage: str,
        system: str,
        user: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        for attempt in range(1, self.max_retries + 2):
            run.log("llm_started", stage=stage, attempt=attempt, backend=self.backend.name)
            try:
                result = self.backend.generate_json(system, user, schema)
                run.log("llm_succeeded", stage=stage, attempt=attempt, backend=self.backend.name)
                return result
            except (TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
                run.log("llm_failed", stage=stage, attempt=attempt, error=f"{type(exc).__name__}: {exc}")
                self._save(run)
                if attempt > self.max_retries:
                    raise
                delay = min(0.25 * 2 ** (attempt - 1), 1.0)
                run.log("retry_scheduled", stage=stage, delay_seconds=delay)
                time.sleep(delay)
        raise RuntimeError("不可达分支")

    @staticmethod
    def _validate_report(report: Dict[str, Any], sentence_count: int) -> None:
        for dimension in ("language", "argumentation"):
            score = float(report["scores"][dimension]["score"])
            if not 1 <= score <= 5:
                raise ValueError(f"{dimension} 分数越界: {score}")
        for item in report.get("highlights", []):
            if not 1 <= int(item["sentence_id"]) <= max(sentence_count, 1):
                raise ValueError("高亮句子编号越界")

    def run(self, prompt: str, essay: str) -> AgentRun:
        if not essay.strip():
            raise ValueError("作文不能为空")
        if not self.rubric_path.is_file():
            raise FileNotFoundError(f"评分量表不存在: {self.rubric_path}")

        run = AgentRun(prompt=prompt, essay=essay)
        run.log("run_started", backend=self.backend.name)
        raw_plan = self._ask(
            run,
            "planning",
            PLANNER_SYSTEM,
            planner_input(prompt, essay, list(self.tools)),
            PLAN_SCHEMA,
        )
        run.plan = self._valid_plan(raw_plan)
        run.log("plan_created", plan=run.plan, backend=self.backend.name)
        self._save(run)

        for step_number, step in enumerate(run.plan, 1):
            tool_name = step["tool"]
            run.log("tool_started", step=step_number, tool=tool_name, reason=step["reason"])
            run.artifacts[tool_name] = self.tools[tool_name](essay, self.rubric_path)
            run.log("tool_succeeded", step=step_number, tool=tool_name)
            self._save(run)

        context = json.dumps(
            {"prompt": prompt, "essay": essay, "tool_results": run.artifacts},
            ensure_ascii=False,
        )
        repair = ""
        for attempt in range(self.max_repairs + 1):
            run.log("llm_report_started", attempt=attempt + 1)
            report = self._ask(
                run,
                "scoring_and_feedback",
                REPORT_SYSTEM,
                context + (f"\n上一轮修复要求：{repair}" if repair else ""),
                REPORT_SCHEMA,
            )
            try:
                self._validate_report(report, len(split_sentences(essay)))
            except Exception as exc:
                run.log("schema_validation_failed", error=str(exc))
                if attempt >= self.max_repairs:
                    raise
                repair = str(exc)
                continue
            reflection = self._ask(
                run,
                "reflection",
                REFLECTOR_SYSTEM,
                json.dumps({"essay": essay, "report": report}, ensure_ascii=False),
                REFLECT_SCHEMA,
            )
            run.log("reflection_completed", **reflection)
            if reflection.get("decision") == "accept" or attempt >= self.max_repairs:
                report["model_backend"] = self.backend.name
                report["run_id"] = run.run_id
                run.report = report
                break
            repair = str(reflection.get("repair_instruction", reflection.get("reason", "请修订")))
        run.log("run_finished", success=run.report is not None)
        self._save(run)
        return run
