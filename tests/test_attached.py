"""A3 — attached mode + hot-reload: a fake Memory client (no network) drives
load, promote-event reload, the preflight canary, and watcher lifecycle."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app
from carl_agent_server.chain_source import MemoryChainSource

from .conftest import SAMPLE_CHAIN, MockLLM


def _record(version: int, content: dict[str, Any], **meta: Any) -> SimpleNamespace:
    return SimpleNamespace(
        entity_id="chain-1",
        version_id=f"vid-{version:04d}aaaa",
        version_number=version,
        channel="stable",
        meta={"display_name": meta.get("display_name", "Weather Agent"), **meta},
        content=content,
    )


class FakeSubscription:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeMemoryClient:
    """Just enough of gigaevo_client.MemoryClient for the agent:
    record reads, watch, and the run-record writes (A6)."""

    def __init__(self, record: SimpleNamespace) -> None:
        self.record = record
        self.callback: Any = None
        self.subscription = FakeSubscription()
        self.fetches = 0
        self.cards: list[dict[str, Any]] = []
        self.runs_recorded: list[tuple[str, str | None]] = []
        self.fail_saves = False

    def get_chain_record(self, entity_id: str, *, channel: str = "latest") -> SimpleNamespace:
        assert entity_id == "chain-1"
        assert channel == "stable"
        self.fetches += 1
        return self.record

    def watch_chain_record(self, entity_id: str, callback: Any, *, channel: str = "latest", **_: Any) -> FakeSubscription:
        self.callback = callback
        return self.subscription

    def promote(self, new_record: SimpleNamespace) -> None:
        """Simulate a promote: repoint the channel, fire the watcher callback."""
        self.record = new_record
        assert self.callback is not None, "watcher was not started"
        self.callback(new_record)

    def save_memory_card(self, memory_card: dict[str, Any], name: str, tags: list[str] | None = None, **_: Any) -> Any:
        if self.fail_saves:
            raise RuntimeError("memory is down")
        self.cards.append({"content": memory_card, "name": name, "tags": tags or []})
        return SimpleNamespace(entity_id="card-1", version_id="cv-1", channel="latest")

    def record_chain_run(self, entity_id: str, run_id: str | None = None) -> Any:
        if self.fail_saves:
            raise RuntimeError("memory is down")
        self.runs_recorded.append((entity_id, run_id))
        return SimpleNamespace(entity_id=entity_id)


def _attached_client(fake: FakeMemoryClient, llm: MockLLM, *, poll_fallback_s: float | None = None) -> TestClient:
    extra = {} if poll_fallback_s is None else {"poll_fallback_s": poll_fallback_s}
    spec = DeploymentSpec(name="weather", entity_id="chain-1", channel="stable", **extra)
    source = MemoryChainSource("chain-1", channel="stable", client=fake)
    return TestClient(build_agent_app(spec, chain_source=source, llm_client=llm))


def _wait_for_version(client: TestClient, expected_prefix: str, deadline_s: float = 5.0) -> dict:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        info = client.get("/info").json()
        if info["version"].startswith(expected_prefix):
            return info
        time.sleep(0.05)
    raise AssertionError(f"version never became {expected_prefix!r}: {client.get('/info').json()['version']}")


def test_attached_load_uses_entity_meta():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN, description="Tells the weather."))
    with _attached_client(fake, MockLLM()) as client:
        info = client.get("/info").json()
        assert info["display_name"] == "Weather Agent"  # entity meta, not the chain name
        assert info["source"] == "memory"
        assert info["channel"] == "stable"
        assert info["version"].startswith("v1")
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "Weather Agent"


def test_promote_event_hot_reloads_and_refreshes_docs():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, MockLLM()) as client:
        assert client.get("/info").json()["version"].startswith("v1")
        new_chain = dict(SAMPLE_CHAIN, name="Echo Researcher v2")
        fake.promote(_record(2, new_chain, display_name="Weather Agent 2.0"))
        info = _wait_for_version(client, "v2")
        assert info["display_name"] == "Weather Agent 2.0"
        # /docs metadata follows the hot-reload
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "Weather Agent 2.0"
        assert schema["info"]["version"].startswith("v2")
        # and the agent still answers
        assert client.post("/invoke", json={"input": "hi"}).json()["status"] == "succeeded"


def test_bad_promotion_keeps_old_chain_serving():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, MockLLM()) as client:
        assert client.get("/readyz").status_code == 200
        broken = dict(SAMPLE_CHAIN)
        broken["steps"] = [
            {
                "step_type": "tool",
                "number": 1,
                "title": "Broken",
                "dependencies": [],
                "step_config": {"tool_name": "no_such_tool", "input_mapping": {}},
            }
        ]
        fake.promote(_record(2, broken))
        time.sleep(0.3)  # give the (failing) reload a moment
        info = client.get("/info").json()
        assert info["version"].startswith("v1")  # canary: old version stayed live
        assert client.get("/readyz").status_code == 200
        assert client.post("/invoke", json={"input": "hi"}).json()["status"] == "succeeded"


def test_watcher_stopped_on_shutdown():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, MockLLM()) as client:
        client.get("/healthz")
        assert fake.callback is not None  # watcher armed on startup
        assert fake.subscription.stopped is False
    assert fake.subscription.stopped is True  # lifespan shutdown stops it


def test_file_source_has_no_watcher(chain_file, mock_llm):
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file))
    app = build_agent_app(spec, llm_client=mock_llm)
    with TestClient(app) as client:
        client.get("/healthz")
        assert app.state.agent._subscription is None  # nothing to watch offline
        assert app.state.agent._poll_task is None  # and nothing to poll


class DeadSSEMemoryClient(FakeMemoryClient):
    """Memory whose events endpoint is broken: the SSE watcher cannot start."""

    def watch_chain_record(self, *args: Any, **kwargs: Any) -> FakeSubscription:
        raise RuntimeError("events endpoint unreachable")


def test_poll_fallback_catches_missed_promote():
    """The 2026-06-11 live bug: SSE subscribed without error, but promote events
    never arrived (the deployed Memory misrouted /v1/events/stream). The poll
    fallback must pick the promote up anyway."""
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, MockLLM(), poll_fallback_s=0.05) as client:
        assert client.get("/info").json()["version"].startswith("v1")
        # promote happens in Memory, but no event reaches the watcher
        fake.record = _record(2, dict(SAMPLE_CHAIN, name="Echo Researcher v2"))
        _wait_for_version(client, "v2")


def test_poll_fallback_reloads_when_sse_cannot_start():
    fake = DeadSSEMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, MockLLM(), poll_fallback_s=0.05) as client:
        assert client.get("/info").json()["version"].startswith("v1")
        fake.record = _record(2, SAMPLE_CHAIN)
        _wait_for_version(client, "v2")


def test_poll_fallback_disabled_serves_stale(mock_llm):
    spec = DeploymentSpec(name="weather", entity_id="chain-1", channel="stable", poll_fallback_s=0)
    fake = DeadSSEMemoryClient(_record(1, SAMPLE_CHAIN))
    source = MemoryChainSource("chain-1", channel="stable", client=fake)
    app = build_agent_app(spec, chain_source=source, llm_client=mock_llm)
    with TestClient(app) as client:
        client.get("/healthz")
        assert app.state.agent._subscription is None  # SSE failed to start
        assert app.state.agent._poll_task is None  # polling explicitly off
        fake.record = _record(2, SAMPLE_CHAIN)
        time.sleep(0.3)
        assert client.get("/info").json()["version"].startswith("v1")  # stale by design


def test_poll_task_stopped_on_shutdown():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    spec = DeploymentSpec(name="weather", entity_id="chain-1", channel="stable", poll_fallback_s=30)
    source = MemoryChainSource("chain-1", channel="stable", client=fake)
    app = build_agent_app(spec, chain_source=source, llm_client=MockLLM())
    with TestClient(app) as client:
        client.get("/healthz")
        assert app.state.agent._poll_task is not None  # armed alongside SSE
    assert app.state.agent._poll_task is None  # lifespan shutdown cancelled it
