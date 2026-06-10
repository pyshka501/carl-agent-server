"""A5 — CLI wiring: both entrypoints build the right app without starting uvicorn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carl_agent_server import cli
from carl_agent_server.hub import AgentHub

from .conftest import SAMPLE_CHAIN


@pytest.fixture()
def captured(monkeypatch):
    calls: dict = {}

    def fake_run(app, **kwargs):
        calls["app"] = app
        calls.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    return calls


def _chain(tmp_path: Path) -> Path:
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(SAMPLE_CHAIN), encoding="utf-8")
    return path


def test_solo_serve_offline(captured, tmp_path):
    chain = _chain(tmp_path)
    cli.main(["serve", "--chain-file", str(chain), "--name", "demo", "--port", "9001"])
    spec = captured["app"].state.agent.spec
    assert spec.name == "demo"
    assert spec.chain_file == str(chain)
    assert captured["port"] == 9001
    assert captured["host"] == "127.0.0.1"


def test_solo_serve_attached(captured):
    cli.main(["serve", "--entity-id", "e-1", "--channel", "latest", "--memory-url", "http://mem:8002"])
    spec = captured["app"].state.agent.spec
    assert spec.entity_id == "e-1"
    assert spec.channel == "latest"
    assert spec.memory_url == "http://mem:8002"
    assert spec.chain_file is None


def test_solo_requires_exactly_one_source(captured):
    with pytest.raises(SystemExit):
        cli.main(["serve"])  # neither --chain-file nor --entity-id


def test_hub_serve(captured, tmp_path):
    state = tmp_path / "state.json"
    cli.hub_main(["serve", "--state-file", str(state), "--port", "9080"])
    hub = captured["app"].state.hub
    assert isinstance(hub, AgentHub)
    assert hub.state_file == state
    assert captured["port"] == 9080


def test_hub_serve_no_persist(captured):
    cli.hub_main(["serve", "--no-persist"])
    assert captured["app"].state.hub.state_file is None
