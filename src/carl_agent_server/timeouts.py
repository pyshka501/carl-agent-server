"""Default-timeout injection for deployed chains (PRODUCTION_TODO C6 / design G9).

Authored chains often omit timeouts. CARL's executor bounds a step by
``step.timeout or chain.timeout`` — so a chain with neither runs every step
unbounded, and a single hung step burns the whole run budget before the
agent's overall deadline (``chain_timeout_s``) even trips.

At load time (cold load AND hot-reload share this one path) we fill in
defaults, **never loosening** what the author already chose:

* **chain-level default** — if the chain has no ``timeout``, set it to
  ``step_timeout_s`` so CARL always has a per-step fallback;
* **per-step default** — every step lacking a ``timeout`` gets
  ``min(step_timeout_s, chain_fallback)``: capped at the global per-step
  default, but never longer than the (authored or injected) chain-level
  timeout. Authored per-step / chain-level values are preserved verbatim.
"""

from __future__ import annotations

import copy
from typing import Any


def inject_default_timeouts(
    content: dict[str, Any], *, step_timeout_s: float
) -> dict[str, Any]:
    """Return a copy of ``content`` with default timeouts filled in.

    Pure: the input dict is not mutated (deep-copied first).
    """
    result = copy.deepcopy(content)

    chain_timeout = result.get("timeout")
    if not isinstance(chain_timeout, int | float) or chain_timeout <= 0:
        chain_timeout = step_timeout_s
        result["timeout"] = chain_timeout

    per_step = min(step_timeout_s, float(chain_timeout))
    steps = result.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            existing = step.get("timeout")
            if not isinstance(existing, int | float) or existing <= 0:
                step["timeout"] = per_step
    return result
