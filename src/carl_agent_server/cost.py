"""Run cost in USD from token usage (PRODUCTION_TODO D4).

Pricing is opt-in and per-deployment (the agent runs ONE model): a pair
``(input_per_1k_usd, output_per_1k_usd)``. Matches CARL's own cost formula —
``cost = prompt/1000 * in + completion/1000 * out`` — and tolerates the usual
token-usage key spellings.
"""

from __future__ import annotations

from typing import Any

_PROMPT_KEYS = ("prompt", "prompt_tokens", "input", "input_tokens")
_COMPLETION_KEYS = ("completion", "completion_tokens", "output", "output_tokens")


def _first(usage: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def run_cost_usd(
    token_usage: dict[str, Any] | None,
    input_per_1k_usd: float | None,
    output_per_1k_usd: float | None,
) -> float | None:
    """USD cost of one run, or ``None`` when pricing isn't configured."""
    if input_per_1k_usd is None or output_per_1k_usd is None:
        return None
    usage = token_usage or {}
    prompt = _first(usage, _PROMPT_KEYS)
    completion = _first(usage, _COMPLETION_KEYS)
    return prompt / 1000.0 * input_per_1k_usd + completion / 1000.0 * output_per_1k_usd
