"""Structured response parsing helpers."""
import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
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
