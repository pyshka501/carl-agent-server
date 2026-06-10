"""Read-only builtin tools: math, time, and the GET/HEAD-only HTTP guard."""

from __future__ import annotations

import pytest

from carl_agent_server.tools import calculator, current_datetime, make_http_request


def test_calculator():
    assert calculator("6*7") == "42"


def test_calculator_rejects_garbage():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('id')")


def test_current_datetime_is_iso():
    from datetime import datetime

    value = current_datetime()
    assert datetime.fromisoformat(value).tzinfo is not None


def test_current_datetime_timezone():
    assert "T" in current_datetime("Europe/Stockholm")


async def test_http_request_blocks_mutating_methods():
    http_request = make_http_request()
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        with pytest.raises(ValueError, match="read-only"):
            await http_request("https://example.com", method=method)
