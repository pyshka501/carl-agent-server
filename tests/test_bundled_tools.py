"""Bundled (deployment-shipped) synthesized tools — registered + run in a subprocess."""

from __future__ import annotations

import asyncio

from carl_agent_server.models import BundledTool, DeploymentSpec
from carl_agent_server.tools import make_bundled_tool, register_bundled_tools


def test_make_bundled_tool_runs_in_subprocess():
    src = "def f(**kwargs):\n    return 'hi ' + kwargs.get('name', '?')"
    out = asyncio.run(make_bundled_tool("f", src)(name="bob"))
    assert out == "hi bob"


def test_bundled_tool_surfaces_its_error_string():
    src = "def f(**kwargs):\n    return 'error: boom'"
    assert asyncio.run(make_bundled_tool("f", src)()) == "error: boom"


def test_bundled_tool_subprocess_crash_is_caught():
    src = "def f(**kwargs):\n    raise RuntimeError('kaboom')"
    out = asyncio.run(make_bundled_tool("f", src)())
    assert out.startswith("error:")  # stderr surfaced, not a hub crash


class _Ctx:
    def __init__(self):
        self.tools: dict = {}

    def register_tool(self, name, fn):
        self.tools[name] = fn


def test_register_bundled_tools():
    ctx = _Ctx()
    names = register_bundled_tools(ctx, [{"name": "f", "source": "def f(**k):\n    return 'x'"}])
    assert names == ["f"] and "f" in ctx.tools


def test_register_bundled_tools_gated_off(monkeypatch):
    monkeypatch.setenv("AGENT_ALLOW_BUNDLED_TOOLS", "0")
    ctx = _Ctx()
    assert register_bundled_tools(ctx, [{"name": "f", "source": "def f(**k):\n    return 'x'"}]) == []
    assert ctx.tools == {}


def test_deployment_spec_accepts_extra_tools():
    spec = DeploymentSpec(
        name="a", entity_id="x",
        extra_tools=[BundledTool(name="f", source="def f(**k):\n    return 'x'")],
    )
    assert [t.name for t in spec.extra_tools] == ["f"]
    # default is empty — backward compatible
    assert DeploymentSpec(name="b", entity_id="y").extra_tools == []
