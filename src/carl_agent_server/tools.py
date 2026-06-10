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

import html as _html
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from simpleeval import simple_eval

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
