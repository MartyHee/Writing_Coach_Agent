"""Lazy Hugging Face local-model adapter."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .json_utils import extract_json


class HuggingFaceJSONBackend:
    """Ask a small instruct model for atomic judgments and assemble a stable report."""

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
            self.model_id, dtype="auto", low_cpu_mem_usage=True
        )
        self._model.eval()

    def _inputs(self, system: str, user: str):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._tokenizer(prompt, return_tensors="pt")

    def _next_token_choice(self, user: str, instruction: str, choices: List[int]) -> int:
        import torch

        inputs = self._inputs(instruction, user + f"\nValid choices: {choices}. Answer one digit only:")
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0, -1]
        strengths = {}
        for choice in choices:
            token_ids = self._tokenizer.encode(str(choice), add_special_tokens=False)
            if token_ids:
                strengths[choice] = float(logits[token_ids[0]])
        if not strengths:
            raise ValueError("model tokenizer cannot encode numeric choices")
        return max(strengths, key=strengths.get)

    def _short_text(self, user: str, instruction: str, max_new_tokens: int = 48) -> str:
        import torch

        inputs = self._inputs(instruction + " Return one concise sentence only.", user)
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

    def _short_fields(self, user: str) -> Dict[str, str]:
        import torch

        instruction = (
            "Return exactly two labelled lines and nothing else.\n"
            "LANGUAGE: one short diagnosis about grammar, clarity, or cohesion.\n"
            "ARGUMENT: one short diagnosis about the claim, reasons, evidence, or counterargument.\n"
            "Use the essay and tool results. Keep each diagnosis concise."
        )
        inputs = self._inputs(instruction, user)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=48, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        fields = {}
        for raw_line in text.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", raw_line.strip())
            match = re.match(r"(?i)^(language|argument(?:ation)?)\s*[:：-]\s*(.+)$", line)
            if match:
                key = "language" if match.group(1).lower() == "language" else "argument"
                fields[key] = match.group(2).strip(' "')
        return {
            "language": fields.get("language", "Language clarity and cohesion can be improved."),
            "argument": fields.get("argument", "The argument needs more specific evidence and explanation."),
        }

    def _evidence_id(self, user: str) -> int:
        try:
            payload = json.loads(user)
            rows = payload.get("tool_results", {}).get("locate_evidence", {}).get("sentence_evidence", [])
            choices = [int(row["sentence_id"]) for row in rows][:9]
        except Exception:
            choices = []
        return self._next_token_choice(user, "Choose the sentence that most needs evidence or explanation.", choices) if choices else 1

    def _plan(self, user: str) -> Dict[str, Any]:
        options = {1: "inspect_text", 2: "load_rubric", 3: "locate_evidence"}
        selected = options[self._next_token_choice(
            user, "Choose the most useful first tool: 1 inspect_text, 2 load_rubric, 3 locate_evidence.", list(options)
        )]
        return {
            "goal": "model-selected grounded diagnosis",
            "steps": [{"tool": selected, "reason": self._short_text(user, f"Explain briefly why {selected} is useful for this task.", 16)}],
            "structured_from_model_fields": True,
        }

    def _report(self, user: str) -> Dict[str, Any]:
        language_score = float(self._next_token_choice(user, "Score language clarity, grammar, and cohesion from 1 weak to 5 strong.", [1, 2, 3, 4, 5]))
        argument_score = float(self._next_token_choice(user, "Score claim, reasons, evidence, and counterargument from 1 weak to 5 strong.", [1, 2, 3, 4, 5]))
        fields = self._short_fields(user)
        evidence_id = self._evidence_id(user)
        language_action = "Rewrite unclear sentences and check grammar, sentence structure, and linking words." if language_score < 3 else "Polish sentence connections and replace vague wording with precise language."
        argument_action = "Add one specific example and explain how it supports the main claim." if argument_score < 3 else "Add a specific example or counterargument and explain how it supports the main claim."
        strengths = []
        if language_score >= 3:
            strengths.append("The draft is generally understandable.")
        if argument_score >= 3:
            strengths.append("The draft presents a recognizable position.")
        if not strengths:
            strengths.append("The draft provides a workable starting point for revision.")
        return {
            "summary": f"Language: {fields['language']}\n\nArgumentation: {fields['argument']}",
            "scores": {
                "language": {"score": language_score, "rationale": fields["language"], "evidence_sentence_ids": [evidence_id]},
                "argumentation": {"score": argument_score, "rationale": fields["argument"], "evidence_sentence_ids": [evidence_id]},
            },
            "strengths": strengths,
            "priorities": [
                {"issue": fields["argument"], "evidence_sentence_id": evidence_id, "action": argument_action, "example": "For example, ... This shows that ..."},
                {"issue": fields["language"], "evidence_sentence_id": evidence_id, "action": language_action, "example": "This sentence is clearer because ..."},
            ],
            "highlights": [{"sentence_id": evidence_id, "label": "needs_evidence", "reason": fields["argument"]}],
            "revision_plan": [argument_action, language_action, "Read the revised paragraph once more and check that each example is explained."],
            "confidence": round((language_score + argument_score) / 10, 2),
            "model_fields": fields,
            "structured_from_model_fields": True,
        }

    def generate_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        self._load()
        task = schema.get("task")
        if task == "plan":
            return self._plan(user)
        if task == "report":
            return self._report(user)
        if task == "reflect":
            score = self._next_token_choice(user, "Rate report grounding and actionability from 1 weak to 5 strong.", [1, 2, 3, 4, 5])
            return {
                "decision": "accept" if score >= 3 else "revise",
                "reason": f"Local model reflection score: {score}/5",
                "repair_instruction": "Improve grounding and actionability." if score < 3 else "",
            }
        inputs = self._inputs(system + "\n只返回一个合法 JSON 对象，不要 Markdown。", user + f"\n输出约束: {json.dumps(schema, ensure_ascii=False)}")
        output = self._model.generate(**inputs, max_new_tokens=64, do_sample=False, repetition_penalty=1.05)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return extract_json(self._tokenizer.decode(generated, skip_special_tokens=True))
