"""Shared fixtures: a sample CARL chain on disk + a mock LLM client.

The mock LLM is the only fake — /invoke tests execute the chain through the
REAL mmar-carl DAG executor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SAMPLE_CHAIN: dict[str, Any] = {
    "name": "Echo Researcher",
    "description": "Answers questions in two reasoning steps.",
    "max_workers": 1,
    "timeout": 60.0,
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Analyze",
            "aim": "Understand the question",
            "reasoning_questions": "What is being asked?",
            "step_context_queries": [],
            "stage_action": "Analyze the question",
            "example_reasoning": "The user asks X, therefore...",
            "dependencies": [],
            "retry_max": 1,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Answer",
            "aim": "Answer the question",
            "reasoning_questions": "What is the final answer?",
            "step_context_queries": [],
            "stage_action": "Write the final answer",
            "example_reasoning": "Based on step 1...",
            "dependencies": [1],
            "retry_max": 1,
        },
    ],
}


class MockLLM:
    """Minimal stub mmar-carl auto-detects as an LLM client (has get_response).

    ``delay`` makes each call slow — used by the cancel/timeout tests.
    """

    def __init__(self, answers: list[str] | None = None, delay: float = 0.0) -> None:
        self.answers = answers or ["step-1 reasoning", "FINAL ANSWER 42"]
        self.delay = delay
        self.calls: list[str] = []

    async def get_response(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        self.calls.append(prompt)
        if self.delay:
            import asyncio

            await asyncio.sleep(self.delay)
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]

    async def get_response_with_retries(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        return await self.get_response(prompt, *args, **kwargs)

    async def get_response_with_system(self, system_prompt: str, prompt: str, *args: Any, **kwargs: Any) -> str:
        return await self.get_response(prompt, *args, **kwargs)

    async def get_response_with_usage(self, prompt: str, *args: Any, **kwargs: Any) -> tuple[str, dict[str, int]]:
        text = await self.get_response(prompt, *args, **kwargs)
        return text, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.fixture()
def chain_file(tmp_path: Path) -> Path:
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(SAMPLE_CHAIN), encoding="utf-8")
    return path


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()
