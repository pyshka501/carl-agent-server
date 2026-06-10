"""A6 — run-records: every finished run writes a CARE-shaped memory_card +
record_chain_run, strictly best-effort (Memory failures never fail the invoke)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app
from carl_agent_server.chain_source import MemoryChainSource

from .conftest import SAMPLE_CHAIN, MockLLM
from .test_attached import FakeMemoryClient, _record


def _attached_client(fake: FakeMemoryClient, llm: MockLLM) -> TestClient:
    spec = DeploymentSpec(name="weather", entity_id="chain-1", channel="stable")
    source = MemoryChainSource("chain-1", channel="stable", client=fake)
    return TestClient(build_agent_app(spec, chain_source=source, llm_client=llm))


def test_successful_run_writes_card_and_run_ping():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, MockLLM()) as client:
        body = client.post("/invoke", json={"input": "what's the weather?"}).json()
        assert body["status"] == "succeeded"
    assert len(fake.cards) == 1
    card = fake.cards[0]
    content = card["content"]
    # CARE run_recorder shape — one replay/history mechanism for TUI and hub runs
    assert content["category"] == "agent_run"
    assert content["task_description"] == "what's the weather?"
    assert content["usage"]["run_id"] == body["run_id"]
    assert content["usage"]["agent_entity_id"] == "chain-1"
    assert content["usage"]["metrics"]["exit_status"] == "success"
    assert content["usage"]["metrics"]["step_count"] == 2
    assert content["usage"]["answer_preview"] == "FINAL ANSWER 42"
    assert "status:success" in card["tags"]
    assert "agent:chain-1" in card["tags"]
    assert "source:agent-server" in card["tags"]
    assert card["name"].startswith("weather · success · ")
    # and the entity's run counters were bumped with the same run id
    assert fake.runs_recorded == [("chain-1", body["run_id"])]


def test_failed_run_is_recorded_with_failed_status():
    class ExplodingLLM(MockLLM):
        async def get_response(self, prompt: str, *args, **kwargs) -> str:
            raise RuntimeError("llm down")

    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    with _attached_client(fake, ExplodingLLM()) as client:
        body = client.post("/invoke", json={"input": "hi"}).json()
        assert body["success"] is False
    assert len(fake.cards) == 1
    assert "status:failed" in fake.cards[0]["tags"]
    assert fake.cards[0]["content"]["usage"]["metrics"]["exit_status"] == "failed"


def test_memory_failure_never_fails_the_invoke():
    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))
    fake.fail_saves = True
    with _attached_client(fake, MockLLM()) as client:
        body = client.post("/invoke", json={"input": "hi"}).json()
        assert body["status"] == "succeeded"  # answering outranks bookkeeping
        assert body["answer"] == "FINAL ANSWER 42"
    assert fake.cards == []
    assert fake.runs_recorded == []


def test_file_agent_records_nothing(chain_file, mock_llm):
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file))
    app = build_agent_app(spec, llm_client=mock_llm)
    with TestClient(app) as client:
        assert client.post("/invoke", json={"input": "hi"}).json()["status"] == "succeeded"
    assert app.state.agent._recorder is None  # offline mode: nothing to record to
