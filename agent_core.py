"""Writing Coach Agent shared by lessons 3/4 and the product app.

The deterministic functions in this module are sensors/tools. Planning, rubric
scoring, feedback, revision planning and reflection are performed by an LLM.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


def _extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from a model response, including fenced output."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    decoder = json.JSONDecoder()
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        break
    raise ValueError(f"模型未返回可解析 JSON: {text[:200]}")


class JSONBackend(Protocol):
    name: str

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class HuggingFaceJSONBackend:
    """Lazy local open-source instruct model; downloads only on first call.

    The small classroom model is asked for short atomic judgments (score,
    evidence ID, rationale, action) and Python assembles the stable contract.
    This keeps semantic decisions model-driven without trusting a 0.5B model
    to reproduce a long nested JSON schema perfectly on every request.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model_id = model_id
        self.name = f"huggingface:{model_id}"
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype="auto",
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def _next_token_choice(self, user: str, instruction: str, choices: list[int]) -> int:
        """Use model logits to choose among explicit integer candidates."""
        import torch

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user + f"\nValid choices: {choices}. Answer one digit only:"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0, -1]
        strengths: dict[int, float] = {}
        for choice in choices:
            token_ids = self._tokenizer.encode(str(choice), add_special_tokens=False)
            if token_ids:
                strengths[choice] = float(logits[token_ids[0]])
        if not strengths:
            raise ValueError("model tokenizer cannot encode numeric choices")
        return max(strengths, key=strengths.get)

    def _short_text(self, user: str, instruction: str, max_new_tokens: int = 48) -> str:
        """Generate one concise semantic field instead of a fragile long JSON blob."""
        import torch

        messages = [
            {"role": "system", "content": instruction + " Return one concise sentence only."},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip().splitlines()[0].strip(' "')
        if not text:
            raise ValueError("local model returned an empty field")
        return text

    def _short_fields(self, user: str) -> dict[str, str]:
        """Generate independent language/argument fields in one short call.

        The model is not asked to produce the full report. It only returns two
        labelled sentences; Python validates the labels and assembles the
        student-facing report below. This keeps the small model fast and avoids
        copying one generic coaching sentence into every report field.
        """
        import torch

        instruction = (
            "Return exactly two labelled lines and nothing else.\n"
            "LANGUAGE: one short diagnosis about grammar, clarity, or cohesion.\n"
            "ARGUMENT: one short diagnosis about the claim, reasons, evidence, or counterargument.\n"
            "Use the essay and tool results. Keep each diagnosis concise."
        )
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        fields: dict[str, str] = {}
        for raw_line in text.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", raw_line.strip())
            match = re.match(r"(?i)^(language|argument(?:ation)?)\s*[:：-]\s*(.+)$", line)
            if not match:
                continue
            key = "language" if match.group(1).lower() == "language" else "argument"
            value = match.group(2).strip(' \"')
            if value:
                fields[key] = value

        # Keep the report contract valid even if the tiny model ignores a label.
        return {
            "language": fields.get("language", "Language clarity and cohesion can be improved."),
            "argument": fields.get("argument", "The argument needs more specific evidence and explanation."),
        }

    def _evidence_id(self, user: str) -> int:
        """Let the model select a valid cited sentence instead of hard-coding ID 1."""
        try:
            payload = json.loads(user)
            rows = payload.get("tool_results", {}).get("locate_evidence", {}).get("sentence_evidence", [])
            choices = [int(row["sentence_id"]) for row in rows][:9]
        except Exception:
            choices = []
        return self._next_token_choice(
            user, "Choose the sentence that most needs evidence or explanation.", choices
        ) if choices else 1

    def _report(self, user: str) -> dict[str, Any]:
        """Build the product report from model-selected atomic fields."""
        language_score = float(self._next_token_choice(
            user, "Score language clarity, grammar, and cohesion from 1 weak to 5 strong.", [1, 2, 3, 4, 5]))
        argument_score = float(self._next_token_choice(
            user, "Score claim, reasons, evidence, and counterargument from 1 weak to 5 strong.", [1, 2, 3, 4, 5]))
        fields = self._short_fields(user)
        language_feedback = fields["language"]
        argument_feedback = fields["argument"]
        evidence_id = self._evidence_id(user)
        language_action = (
            "Rewrite unclear sentences and check grammar, sentence structure, and linking words."
            if language_score < 3
            else "Polish sentence connections and replace any vague wording with precise language."
        )
        argument_action = (
            "Add one specific example and explain how it supports the main claim."
            if argument_score < 3
            else "Add a specific example or counterargument and explain how it supports the main claim."
        )
        strengths = []
        if language_score >= 3:
            strengths.append("The draft is generally understandable.")
        if argument_score >= 3:
            strengths.append("The draft presents a recognizable position.")
        if not strengths:
            strengths.append("The draft provides a workable starting point for revision.")
        return {
            "summary": f"Language: {language_feedback}\n\nArgumentation: {argument_feedback}",
            "scores": {
                "language": {"score": language_score, "rationale": language_feedback,
                             "evidence_sentence_ids": [evidence_id]},
                "argumentation": {"score": argument_score, "rationale": argument_feedback,
                                  "evidence_sentence_ids": [evidence_id]},
            },
            "strengths": strengths,
            "priorities": [
                {"issue": argument_feedback, "evidence_sentence_id": evidence_id,
                 "action": argument_action, "example": "For example, ... This shows that ..."},
                {"issue": language_feedback, "evidence_sentence_id": evidence_id,
                 "action": language_action, "example": "This sentence is clearer because ..."},
            ],
            "highlights": [{"sentence_id": evidence_id, "label": "needs_evidence",
                            "reason": argument_feedback}],
            "revision_plan": [
                argument_action,
                language_action,
                "Read the revised paragraph once more and check that each example is explained.",
            ],
            "confidence": round((language_score + argument_score) / 10, 2),
            "model_fields": fields,
            "structured_from_model_fields": True,
        }

    def _plan(self, user: str) -> dict[str, Any]:
        """Model-select one first tool; the Executor may add grounding tools."""
        options = {1: "inspect_text", 2: "load_rubric", 3: "locate_evidence"}
        selected_number = self._next_token_choice(
            user,
            "Choose the most useful first tool: 1 inspect_text, 2 load_rubric, 3 locate_evidence.",
            list(options),
        )
        selected = options[selected_number]
        reason = self._short_text(user, f"Explain briefly why {selected} is useful for this task.", 16)
        return {"goal": "model-selected grounded diagnosis",
                "steps": [{"tool": selected, "reason": reason}],
                "structured_from_model_fields": True}

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        self._load()
        task = schema.get("task")
        if task == "plan":
            return self._plan(user)
        if task == "report":
            return self._report(user)
        if task == "reflect":
            score = self._next_token_choice(
                user, "Rate report grounding and actionability from 1 weak to 5 strong.", [1, 2, 3, 4, 5])
            return {"decision": "accept" if score >= 3 else "revise",
                    "reason": f"Local model reflection score: {score}/5",
                    "repair_instruction": "Improve grounding and actionability." if score < 3 else ""}
        schema_text = json.dumps(schema, ensure_ascii=False)
        messages = [
            {"role": "system", "content": system + "\n只返回一个合法 JSON 对象，不要 Markdown。"},
            {"role": "user", "content": user + f"\n输出约束: {schema_text}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        output = self._model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.05,
        )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return _extract_json(self._tokenizer.decode(generated, skip_special_tokens=True))


class DemoJSONBackend:
    """Offline classroom fallback. It is deliberately labelled non-AI."""

    name = "offline-demo-rules (not AI)"
    used_fallback = True

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        task = schema.get("task")
        if task == "plan":
            return {"goal": "根据量表和作文证据完成诊断", "steps": [
                {"tool": "inspect_text", "reason": "获取可验证的文本特征"},
                {"tool": "load_rubric", "reason": "读取评分标准"},
                {"tool": "locate_evidence", "reason": "定位可引用句子"},
            ]}
        if task == "report":
            essay_match = re.search(r'"essay"\s*:\s*"(.*?)"', user, re.S)
            essay = essay_match.group(1) if essay_match else user
            sentences = split_sentences(essay)
            wc = len(re.findall(r"\b[A-Za-z']+\b", essay))
            has_example = bool(re.search(r"\b(for example|for instance|such as)\b", essay, re.I))
            has_counter = bool(re.search(r"\b(however|although|some people)\b", essay, re.I))
            language = min(5.0, round(1.5 + wc / 45, 1))
            argument = min(5.0, round(1.4 + has_example + has_counter + min(wc / 100, 1), 1))
            evidence_id = 1 if sentences else 0
            return {
                "summary": "立场可识别，但论证链还需要更具体的证据与解释。",
                "scores": {
                    "language": {"score": language, "rationale": "根据清晰度、词汇和句式综合判断。", "evidence_sentence_ids": [evidence_id]},
                    "argumentation": {"score": argument, "rationale": "根据主张、理由、例证和反方回应综合判断。", "evidence_sentence_ids": [evidence_id]},
                },
                "strengths": ["已经表达了可识别的立场。"],
                "priorities": [{"issue": "论证展开不足", "evidence_sentence_id": evidence_id,
                    "action": "选一个理由，补充‘具体事例—这说明什么’两句。",
                    "example": "For example, ... This shows that ..."}],
                "highlights": [{"sentence_id": evidence_id, "label": "needs_evidence", "reason": "该句适合继续补充证据。"}],
                "revision_plan": ["补充一个具体例子", "解释例子如何支持主张", "检查句子连接"],
                "confidence": 0.55,
            }
        if task == "reflect":
            return {"decision": "accept", "reason": "报告字段完整且引用了句子编号。", "repair_instruction": ""}
        if task == "evaluate":
            return {"pass": True, "score": 4, "reason": "建议包含问题、文本证据与可执行动作。", "risks": ["当前为离线演示后端"]}
        raise ValueError(f"未知演示任务: {task}")


class FallbackBackend:
    def __init__(self, primary: JSONBackend, fallback: JSONBackend) -> None:
        self.primary, self.fallback = primary, fallback
        self.name = primary.name
        self.last_error: str | None = None
        self.used_fallback = False

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.primary.generate_json(system, user, schema)
            self.name = self.primary.name
            return result
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.used_fallback = True
            self.name = self.fallback.name
            return self.fallback.generate_json(system, user, schema)


def create_backend(allow_offline_fallback: bool = True) -> JSONBackend:
    model_id = os.getenv("WRITING_COACH_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    primary: JSONBackend = HuggingFaceJSONBackend(model_id)
    return FallbackBackend(primary, DemoJSONBackend()) if allow_offline_fallback else primary


def split_sentences(essay: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", essay.strip()) if s.strip()]


def inspect_text(essay: str) -> dict[str, Any]:
    sentences = split_sentences(essay)
    words = re.findall(r"\b[A-Za-z']+\b", essay)
    return {"word_count": len(words), "sentence_count": len(sentences),
            "sentences": [{"id": i + 1, "text": s} for i, s in enumerate(sentences)]}


def load_rubric(rubric_path: Path) -> dict[str, Any]:
    return json.loads(rubric_path.read_text(encoding="utf-8"))


def locate_evidence(essay: str) -> dict[str, Any]:
    patterns = {
        "claim": r"\b(should|must|believe|opinion)\b",
        "reason": r"\b(because|since|reason)\b",
        "example": r"\b(for example|for instance|such as)\b",
        "counterargument": r"\b(however|although|some people|on the other hand)\b",
        "conclusion": r"\b(therefore|in conclusion|to conclude)\b",
    }
    found = []
    for i, sentence in enumerate(split_sentences(essay), 1):
        labels = [label for label, pattern in patterns.items() if re.search(pattern, sentence, re.I)]
        found.append({"sentence_id": i, "labels": labels, "text": sentence})
    return {"sentence_evidence": found}


TOOLS = {
    "inspect_text": lambda essay, rubric_path: inspect_text(essay),
    "load_rubric": lambda essay, rubric_path: load_rubric(rubric_path),
    "locate_evidence": lambda essay, rubric_path: locate_evidence(essay),
}


@dataclass
class AgentRun:
    prompt: str
    essay: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    plan: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None

    def log(self, event: str, **details: Any) -> None:
        self.trace.append({"time": time.strftime("%H:%M:%S"), "event": event, **details})


PLAN_SCHEMA = {"task": "plan", "goal": "string", "steps": [{"tool": "one allowed tool", "reason": "string"}]}
REPORT_SCHEMA = {"task": "report", "summary": "string", "scores": {
    "language": {"score": "1..5", "rationale": "string", "evidence_sentence_ids": [1]},
    "argumentation": {"score": "1..5", "rationale": "string", "evidence_sentence_ids": [1]}},
    "strengths": ["string"], "priorities": [{"issue": "string", "evidence_sentence_id": 1,
    "action": "string", "example": "short fragment, do not rewrite essay"}],
    "highlights": [{"sentence_id": 1, "label": "strength|needs_evidence|language|counterargument", "reason": "string"}],
    "revision_plan": ["ordered action"], "confidence": "0..1"}
REFLECT_SCHEMA = {"task": "reflect", "decision": "accept|revise", "reason": "string", "repair_instruction": "string"}


class WritingCoachAgent:
    def __init__(self, backend: JSONBackend, rubric_path: Path, checkpoint_dir: Path | None = None,
                 max_repairs: int = 1, max_retries: int = 2) -> None:
        self.backend = backend
        self.rubric_path = rubric_path
        self.checkpoint_dir = checkpoint_dir
        self.max_repairs = max_repairs
        self.max_retries = max_retries

    def _save(self, run: AgentRun) -> None:
        if not self.checkpoint_dir:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / f"{run.run_id}.json").write_text(
            json.dumps(asdict(run), ensure_ascii=False, indent=2), encoding="utf-8")
        run.log("checkpoint_saved")

    def _valid_plan(self, plan: dict[str, Any]) -> list[dict[str, str]]:
        valid = []
        seen = set()
        for step in plan.get("steps", []):
            tool = step.get("tool")
            if tool in TOOLS and tool not in seen:
                valid.append({"tool": tool, "reason": str(step.get("reason", ""))})
                seen.add(tool)
        if not valid:
            raise ValueError("Planner 没有选择任何合法工具")
        # The model chooses the first useful tool. The Executor enforces the
        # non-negotiable grounding contract before any student-facing score.
        for required, reason in [
            ("load_rubric", "Executor guardrail: scoring requires the rubric"),
            ("locate_evidence", "Executor guardrail: feedback requires valid sentence evidence"),
        ]:
            if required not in seen:
                valid.append({"tool": required, "reason": reason})
                seen.add(required)
        return valid

    def _ask(self, run: AgentRun, stage: str, system: str, user: str,
             schema: dict[str, Any]) -> dict[str, Any]:
        """Retry transient/model-format failures and expose every attempt in Trace."""
        for attempt in range(1, self.max_retries + 2):
            run.log("llm_started", stage=stage, attempt=attempt, backend=self.backend.name)
            try:
                result = self.backend.generate_json(system, user, schema)
                run.log("llm_succeeded", stage=stage, attempt=attempt, backend=self.backend.name)
                return result
            except (TimeoutError, ConnectionError, RuntimeError, ValueError) as exc:
                run.log("llm_failed", stage=stage, attempt=attempt,
                        error=f"{type(exc).__name__}: {exc}")
                self._save(run)
                if attempt > self.max_retries:
                    raise
                delay = min(0.25 * 2 ** (attempt - 1), 1.0)
                run.log("retry_scheduled", stage=stage, delay_seconds=delay)
                time.sleep(delay)
        raise RuntimeError("不可达分支")

    def _validate_report(self, report: dict[str, Any], sentence_count: int) -> None:
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
        run = AgentRun(prompt=prompt, essay=essay)
        run.log("run_started", backend=self.backend.name)
        plan_raw = self._ask(run, "planning",
            "你是 Writing Coach Planner。根据任务自主选择工具，不要提前打分。",
            f"任务：评估论证文并给出可验证的修订计划。\n题目：{prompt}"
            f"\n作文摘要：{essay[:600]}\n可用工具：{list(TOOLS)}",
            PLAN_SCHEMA,
        )
        run.plan = self._valid_plan(plan_raw)
        run.log("plan_created", plan=run.plan, backend=self.backend.name)
        self._save(run)

        for step_no, step in enumerate(run.plan, 1):
            tool = step["tool"]
            run.log("tool_started", step=step_no, tool=tool, reason=step["reason"])
            run.artifacts[tool] = TOOLS[tool](essay, self.rubric_path)
            run.log("tool_succeeded", step=step_no, tool=tool)
            self._save(run)

        context = json.dumps({"prompt": prompt, "essay": essay, "tool_results": run.artifacts}, ensure_ascii=False)
        repair = ""
        for attempt in range(self.max_repairs + 1):
            run.log("llm_report_started", attempt=attempt + 1)
            report = self._ask(run, "scoring_and_feedback",
                "你是严格但支持学生的论证写作评估 Agent。分数必须基于 rubric 和原文句子证据；建议必须指向问题、引用句号、给出动作，但不代写全文。",
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
            reflection = self._ask(run, "reflection",
                "你是独立 Reflector。检查报告是否严格使用量表、原文证据和可执行建议。",
                json.dumps({"essay": essay, "report": report}, ensure_ascii=False),
                REFLECT_SCHEMA,
            )
            run.log("reflection_completed", **reflection)
            if reflection.get("decision") == "accept" or attempt >= self.max_repairs:
                report["model_backend"] = self.backend.name
                report["degraded"] = getattr(self.backend, "used_fallback", False)
                report["run_id"] = run.run_id
                run.report = report
                break
            repair = str(reflection.get("repair_instruction", reflection.get("reason", "请修订")))
        run.log("run_finished", success=run.report is not None)
        self._save(run)
        return run


LABEL_STYLE = {
    "strength": ("#dcfce7", "#166534", "亮点"),
    "needs_evidence": ("#fef3c7", "#92400e", "需补证据"),
    "language": ("#fee2e2", "#991b1b", "语言"),
    "counterargument": ("#e0e7ff", "#3730a3", "反方回应"),
}


def render_highlighted_essay(essay: str, report: dict[str, Any]) -> str:
    by_id = {int(x["sentence_id"]): x for x in report.get("highlights", [])}
    pieces = ['<div class="essay-paper">']
    for sentence_id, sentence in enumerate(split_sentences(essay), 1):
        mark = by_id.get(sentence_id)
        if mark:
            bg, color, label = LABEL_STYLE.get(mark.get("label"), ("#f1f5f9", "#334155", "关注"))
            title = html.escape(str(mark.get("reason", "")), quote=True)
            pieces.append(f'<span class="sentence-mark" title="{title}" style="background:{bg};color:{color}">{html.escape(sentence)}<small>{label}</small></span> ')
        else:
            pieces.append(f'<span>{html.escape(sentence)}</span> ')
    pieces.append("</div>")
    return "".join(pieces)
