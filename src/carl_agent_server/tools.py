"""Read-only builtin tools for deployed agents.

The deployed tool surface is deliberately READ-ONLY (owner decision,
2026-06-10: current demos ship tools that cannot mutate anything): no
run_python, no file writes, HTTP restricted to GET/HEAD. Mutating tools return
together with a per-deployment tool policy (P2 in PRODUCTION_TODO.md).

Tool failures RAISE (never "error as data" — the masked-tool-error incident
class from the 09.06 audit), so CARL records the step as failed and its
retry/replan machinery can react.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from simpleeval import simple_eval

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_READ_METHODS = frozenset({"GET", "HEAD"})


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression (safe evaluator, no names or calls)."""
    try:
        return str(simple_eval(expression))
    except Exception as exc:
        raise ValueError(f"cannot evaluate {expression!r}: {exc}") from exc


def current_datetime(timezone: str = "UTC") -> str:
    """Current date/time as ISO-8601, optionally in a named timezone."""
    tz = UTC if not timezone or timezone.upper() == "UTC" else ZoneInfo(timezone)
    return datetime.now(tz).isoformat(timespec="seconds")


def _strip_html(text: str) -> str:
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", text))).strip()


def make_fetch_url(max_chars: int = 8000) -> Callable[[str], Awaitable[str]]:
    async def fetch_url(url: str) -> str:
        """Fetch a URL (GET) and return readable text (HTML stripped, truncated)."""
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text
            content_type = resp.headers.get("content-type", "")
        if "html" in content_type:
            text = _strip_html(text)
        return text[:max_chars]

    return fetch_url


def make_http_request(max_chars: int = 8000) -> Callable[..., Awaitable[str]]:
    async def http_request(
        url: str,
        method: str = "GET",
        headers: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Read-only HTTP request. Deployed agents allow GET/HEAD only."""
        normalized = (method or "GET").upper()
        if normalized not in _READ_METHODS:
            raise ValueError(f"method {normalized} is not allowed in a deployed agent (read-only: GET/HEAD)")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.request(normalized, url, headers=headers, params=params)
            return resp.text[:max_chars]

    return http_request


def make_web_search(api_key: str, max_results: int = 5) -> Callable[[str], Awaitable[str]]:
    async def web_search(query: str) -> str:
        """Search the web (Tavily) and return condensed results."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
        lines = [
            f"- {item.get('title', '')} — {item.get('url', '')}\n  {str(item.get('content', ''))[:300]}"
            for item in data.get("results", [])[:max_results]
        ]
        return "\n".join(lines) or "no results"

    return web_search


def register_builtin_tools(context: Any, *, web_search_api_key: str | None = None) -> list[str]:
    """Register the read-only builtin set on a ReasoningContext; returns the tool names."""
    context.register_tool("calculator", calculator)
    context.register_tool("current_datetime", current_datetime)
    context.register_tool("fetch_url", make_fetch_url())
    context.register_tool("http_request", make_http_request())
    names = ["calculator", "current_datetime", "fetch_url", "http_request"]
    if web_search_api_key:
        context.register_tool("web_search", make_web_search(web_search_api_key))
        names.append("web_search")
    return names


# --- bundled (deployment-shipped) synthesized tools -------------------------

_BUNDLED_TOOL_TIMEOUT_S = 30.0


def make_bundled_tool(name: str, source: str, *, max_chars: int = 8000) -> Callable[..., Awaitable[str]]:
    """Wrap a shipped synthesized-tool ``source`` as an async tool.

    Each call runs ``source`` in a FRESH subprocess (a new interpreter), feeding
    kwargs as JSON on stdin and reading the function's return from stdout — so a
    deployed agent can call a tool the hub doesn't ship without exec'ing untrusted
    code inside the long-lived hub process. Not a security sandbox (the subprocess
    shares the host), but it gives process isolation + a hard timeout."""

    async def _tool(**kwargs: Any) -> str:
        runner = (
            source
            + "\n\nif __name__ == '__main__':\n"
            "    import json as _j, sys as _sys\n"
            "    _a = _j.loads(_sys.stdin.read() or '{}')\n"
            f"    _r = {name}(**_a)\n"
            "    _sys.stdout.write(_r if isinstance(_r, str) "
            "else _j.dumps(_r, ensure_ascii=False, default=str))\n"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", runner,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(
                proc.communicate(json.dumps(kwargs, ensure_ascii=False, default=str).encode()),
                timeout=_BUNDLED_TOOL_TIMEOUT_S,
            )
        except TimeoutError:
            return f"error: bundled tool '{name}' timed out after {_BUNDLED_TOOL_TIMEOUT_S:.0f}s"
        except Exception as exc:  # noqa: BLE001
            return f"error: bundled tool '{name}' failed to launch: {exc}"
        text = out.decode("utf-8", "replace").strip()
        if not text and proc.returncode:
            text = "error: " + (err.decode("utf-8", "replace").strip() or f"exit {proc.returncode}")
        return text[:max_chars]

    return _tool


def register_bundled_tools(context: Any, extra_tools: list[Any]) -> list[str]:
    """Register deployment-shipped synthesized tools (``DeploymentSpec.extra_tools``).

    Gated by ``AGENT_ALLOW_BUNDLED_TOOLS`` (default on) — set it to 0/false to
    refuse running shipped code. Returns the names registered."""
    if not extra_tools:
        return []
    allow = os.environ.get("AGENT_ALLOW_BUNDLED_TOOLS", "1").strip().lower()
    if allow in ("0", "false", "no", "off"):
        logger.warning(
            "deployment ships %d bundled tool(s) but AGENT_ALLOW_BUNDLED_TOOLS is off — skipping",
            len(extra_tools),
        )
        return []
    registered: list[str] = []
    for t in extra_tools:
        name = getattr(t, "name", None) if not isinstance(t, dict) else t.get("name")
        source = getattr(t, "source", None) if not isinstance(t, dict) else t.get("source")
        if not name or not source:
            continue
        try:
            context.register_tool(str(name), make_bundled_tool(str(name), str(source)))
            registered.append(str(name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to register bundled tool %r: %s", name, exc)
    if registered:
        logger.warning(
            "registered %d BUNDLED tool(s) (shipped with the deployment, run in subprocess): %s",
            len(registered), ", ".join(registered),
        )
    return registered
