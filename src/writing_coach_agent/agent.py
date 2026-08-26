"""Planner-executor-reflector orchestration."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .backends.base import FallbackCapableJSONBackend, JSONBackend
from .checkpoints import CheckpointStore
from .contracts import PLAN_SCHEMA, REFLECT_SCHEMA, REPORT_SCHEMA
from .models import AgentRun
from .prompts import PLANNER_SYSTEM, REFLECTOR_SYSTEM, REPORT_SYSTEM, planner_input
from .reflection import ExecutionReflector
from .tools import TOOLS, Tool, ToolRegistry, split_sentences


class WritingCoachAgent:
    """Deep module exposing one operation: produce a grounded coaching run."""

    def __init__(
        self,
        backend: JSONBackend,
        rubric_path: Path,
        checkpoint_dir: Optional[Path] = None,
        max_repairs: int = 1,
        max_retries: int = 2,
        max_tool_retries: int = 1,
        max_replans: int = 1,
        max_tool_calls: int = 10,
        tools: Optional[Dict[str, Tool]] = None,
        execution_reflector: Optional[ExecutionReflector] = None,
    ) -> None:
        self.backend = backend
        self.rubric_path = rubric_path
        self.checkpoints = CheckpointStore(checkpoint_dir) if checkpoint_dir else None
        self.max_repairs = max_repairs
        self.max_retries = max_retries
        self.max_tool_retries = max_tool_retries
        self.max_replans = max_replans
        self.max_tool_calls = max_tool_calls
        self.tools = tools or TOOLS
        self.tool_registry = ToolRegistry(self.tools)
        self.execution_reflector = execution_reflector or ExecutionReflector()

    def _activate_fallback(self, run: AgentRun, reason: str) -> bool:
        if not isinstance(self.backend, FallbackCapableJSONBackend) or self.backend.degraded:
            return False
        self.backend.activate_fallback(reason)
        run.memory.remember("fallback_activated", reason=reason, backend=self.backend.name)
        run.log("fallback_activated", reason=reason, backend=self.backend.name)
        self._save(run)
        return True

    def _save(self, run: AgentRun) -> None:
        if self.checkpoints:
            self.checkpoints.save(run)
            run.log("checkpoint_saved")

    def _valid_plan(
        self,
        plan: Dict[str, Any],
        excluded_tools: Optional[set[str]] = None,
        allow_guardrail_only: bool = False,
    ) -> List[Dict[str, str]]:
        valid = []
        seen = set()
        excluded_tools = excluded_tools or set()
        for step in plan.get("steps", []):
            tool = step.get("tool")
            if tool in self.tools and tool not in seen and tool not in excluded_tools:
                valid.append({"tool": tool, "reason": str(step.get("reason", ""))})
                seen.add(tool)
        if not valid and not allow_guardrail_only:
            raise ValueError("Planner 没有选择任何合法工具")
        required_tools = [("load_rubric", "Executor guardrail: scoring requires the rubric")]
        evidence_tool = "retrieve_evidence" if "retrieve_evidence" in self.tools and "retrieve_evidence" not in excluded_tools else "locate_evidence"
        required_tools.append((evidence_tool, "Executor guardrail: feedback requires valid sentence evidence"))
        for required, reason in required_tools:
            if required not in seen:
                if required not in self.tools:
                    raise ValueError(f"Executor 缺少必需工具: {required}")
                valid.append({"tool": required, "reason": reason})
                seen.add(required)
        if not valid:
            raise ValueError("Planner 没有选择任何合法工具")
        return valid

    def _replan(
        self,
        run: AgentRun,
        failed_tools: set[str],
        replan_number: int,
    ) -> List[Dict[str, str]]:
        run.log("replan_started", replan=replan_number, failed_tools=sorted(failed_tools))
        user = (
            planner_input(run.prompt, run.essay, self.tool_registry.names)
            + "\nPrevious execution failed. Do not select these tools: "
            + json.dumps(sorted(failed_tools), ensure_ascii=False)
            + "\nMemory: "
            + json.dumps(run.memory.context(), ensure_ascii=False)
        )
        raw_plan = self._ask(run, "replanning", PLANNER_SYSTEM, user, PLAN_SCHEMA)
        plan = self._valid_plan(raw_plan, excluded_tools=failed_tools, allow_guardrail_only=True)
        run.plan.extend(plan)
        run.memory.remember_plan(plan, reason=f"replan_{replan_number}")
        run.log("replan_created", replan=replan_number, plan=plan)
        self._save(run)
        return plan

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
                    reason = f"{stage} retries exhausted: {type(exc).__name__}: {exc}"
                    if self._activate_fallback(run, reason):
                        run.log("llm_started", stage=stage, attempt="fallback", backend=self.backend.name)
                        result = self.backend.generate_json(system, user, schema)
                        run.log("llm_succeeded", stage=stage, attempt="fallback", backend=self.backend.name)
                        return result
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
        if isinstance(self.backend, FallbackCapableJSONBackend):
            self.backend.reset()
        run.log("run_started", backend=self.backend.name)
        raw_plan = self._ask(
            run,
            "planning",
            PLANNER_SYSTEM,
            planner_input(prompt, essay, self.tool_registry.names),
            PLAN_SCHEMA,
        )
        try:
            run.plan = self._valid_plan(raw_plan)
        except (KeyError, TypeError, ValueError) as exc:
            reason = f"plan schema validation failed: {type(exc).__name__}: {exc}"
            run.log("schema_validation_failed", stage="planning", error=reason)
            if not self._activate_fallback(run, reason):
                raise
            fallback_plan = self._ask(run, "planning_fallback", PLANNER_SYSTEM, planner_input(prompt, essay, self.tool_registry.names), PLAN_SCHEMA)
            run.plan = self._valid_plan(fallback_plan)
            run.log("plan_schema_fallback_succeeded", backend=self.backend.name)
        run.memory.remember_plan(run.plan)
        run.log("plan_created", plan=run.plan, backend=self.backend.name)
        self._save(run)

        pending = list(run.plan)
        failed_tools: set[str] = set()
        tool_calls = 0
        replans = 0
        while pending:
            step = pending.pop(0)
            tool_name = step["tool"]
            if tool_name in run.artifacts:
                run.log("tool_skipped", tool=tool_name, reason="result already present in Memory")
                continue
            attempt = 0
            while True:
                attempt += 1
                tool_calls += 1
                if tool_calls > self.max_tool_calls:
                    raise RuntimeError("达到最大工具调用次数")
                run.log("tool_started", tool=tool_name, reason=step["reason"], attempt=attempt)
                try:
                    result = self.tool_registry.call(tool_name, essay, self.rubric_path)
                    run.artifacts[tool_name] = result
                    run.memory.remember_tool_result(tool_name, result)
                    run.log("tool_succeeded", tool=tool_name, attempt=attempt)
                    self._save(run)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    decision = self.execution_reflector.review_failure(
                        tool_name, exc, attempt, self.max_tool_retries
                    )
                    run.memory.remember_failure(tool_name, error, decision.action)
                    run.log(
                        "tool_failed",
                        tool=tool_name,
                        attempt=attempt,
                        error=error,
                        decision=decision.action,
                        decision_reason=decision.reason,
                    )
                    self._save(run)
                    if decision.action == "retry":
                        continue
                    if decision.action == "replan" and replans < self.max_replans:
                        failed_tools.add(tool_name)
                        replans += 1
                        pending = self._replan(run, failed_tools, replans) + pending
                        break
                    raise RuntimeError(f"工具执行失败且无法继续: {tool_name}: {error}") from exc

        context = json.dumps(
            {
                "prompt": prompt,
                "essay": essay,
                "tool_results": run.artifacts,
                "memory": run.memory.context(),
            },
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
                    reason = f"schema validation failed: report missing or invalid fields: {type(exc).__name__}: {exc}"
                    if not self._activate_fallback(run, reason):
                        raise
                    report = self._ask(
                        run,
                        "scoring_and_feedback_fallback",
                        REPORT_SYSTEM,
                        context,
                        REPORT_SCHEMA,
                    )
                    self._validate_report(report, len(split_sentences(essay)))
                    run.log("schema_fallback_succeeded", backend=self.backend.name)
                else:
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
            run.memory.remember("report_reflected", **reflection)
            if reflection.get("decision") == "accept" or attempt >= self.max_repairs:
                report["model_backend"] = self.backend.name
                report["degraded"] = bool(getattr(self.backend, "degraded", False))
                report["fallback_reason"] = getattr(self.backend, "fallback_reason", None)
                report["run_id"] = run.run_id
                run.report = report
                break
            repair = str(reflection.get("repair_instruction", reflection.get("reason", "请修订")))
        run.log("run_finished", success=run.report is not None)
        self._save(run)
        return run
