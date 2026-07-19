"""Canonical chat prompt construction for Stage 2 datasets."""

from __future__ import annotations

from typing import Any, Mapping


def system_message(config: Mapping[str, Any]) -> dict[str, str]:
    """Build the shared system message from config text."""

    return {"role": "system", "content": str(config["preprocessing"]["system_prompt"]).strip()}


def user_message(problem: str, config: Mapping[str, Any]) -> dict[str, str]:
    """Build the shared user message from config text and the math problem."""

    instruction = str(config["preprocessing"]["user_instruction"]).strip()
    return {"role": "user", "content": f"{instruction}\n\nProblem:\n{problem.strip()}"}


def assistant_message(content: str) -> dict[str, str]:
    """Build an assistant message."""

    return {"role": "assistant", "content": content.strip()}


def base_messages(problem: str, config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the system and user messages shared by SFT and DPO."""

    return [system_message(config), user_message(problem, config)]
