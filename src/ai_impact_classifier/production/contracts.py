"""Shared target and text contracts for the production AI-impact model."""
from __future__ import annotations

import re
from typing import Final


MODEL_VERSION: Final[str] = "gpt52-current-guidance-e0-e1-e23-v1"
CLASS_ORDER: Final[tuple[str, ...]] = ("E0", "E1", "E23")
SOURCE_TO_TARGET: Final[dict[str, str]] = {
    "E0": "E0",
    "E1": "E1",
    "E2": "E23",
    "E3": "E23",
}
_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-zA-Z]{2,}")


def normalize_text(value: object) -> str:
    """Normalize spacing only; semantic wording must remain available to the model."""
    return _WHITESPACE.sub(" ", str(value or "").strip())


def compose_model_text(
    task: object,
    jobrole_title: object | None = None,
    action: object | None = None,
    object_: object | None = None,
    purpose: object | None = None,
) -> str:
    task_text = normalize_text(task)
    title_text = normalize_text(jobrole_title)
    if not task_text:
        raise ValueError("keytaskContent must not be blank")
    parts = [f"[TASK] {task_text}"]
    if title_text:
        parts.insert(0, f"[TITLE] {title_text}")
    for name, value in (("ACTION", action), ("OBJECT", object_), ("PURPOSE", purpose)):
        value_text = normalize_text(value)
        if value_text:
            parts.append(f"[{name}] {value_text}")
    return " ".join(parts)


def compose_task_from_aop(action: object, object_: object, purpose: object | None = None) -> str:
    """Create the canonical task field used in training from supplied AOP parts."""
    task = normalize_text(" ".join(part for part in (action, object_, purpose) if normalize_text(part)))
    if not task:
        raise ValueError("action and object must produce a non-blank AOP task")
    return task


def lexical_tokens(value: object) -> list[str]:
    return _TOKEN.findall(normalize_text(value).lower())
