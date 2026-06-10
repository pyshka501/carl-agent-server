"""A2 — async invoke (202 + poll), SSE step events, cooperative cancel, timeout."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app

from .conftest import MockLLM


def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def _poll_until_terminal(client: TestClient, run_id: str, deadline_s: float = 10.0) -> dict:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {deadline_s}s")


def test_async_invoke_returns_202_and_completes(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        response = client.post("/invoke?mode=async", json={"input": "hi"})
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "running"
        final = _poll_until_terminal(client, body["run_id"])
        assert final["status"] == "succeeded"
        assert final["answer"] == "FINAL ANSWER 42"
        assert len(final["steps"]) == 2


def test_sse_streams_steps_then_result(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        run_id = client.post("/invoke?mode=async", json={"input": "hi"}).json()["run_id"]
        events: list[str] = []
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            assert stream.headers["content-type"].startswith("text/event-stream")
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    events.append(line.removeprefix("event: "))
        # full replay: one event per chain step, then the terminal result
        assert events == ["step", "step", "result"]


def test_sse_replays_for_late_subscriber(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        run_id = client.post("/invoke?mode=async", json={"input": "hi"}).json()["run_id"]
        _poll_until_terminal(client, run_id)  # subscribe only AFTER the run finished
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            events = [line.removeprefix("event: ") for line in stream.iter_lines() if line.startswith("event: ")]
        assert events == ["step", "step", "result"]


def test_sse_unknown_run_404(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert client.get("/runs/nope/events").status_code == 404


def test_delete_cancels_a_running_run(chain_file):
    slow = MockLLM(delay=0.4)
    with _client(chain_file, slow) as client:
        run_id = client.post("/invoke?mode=async", json={"input": "hi"}).json()["run_id"]
        response = client.delete(f"/runs/{run_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelling"
        final = _poll_until_terminal(client, run_id)
        assert final["status"] == "cancelled"
        assert final["success"] is False


def test_delete_finished_run_conflicts(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        body = client.post("/invoke", json={"input": "hi"}).json()  # sync — finished
        response = client.delete(f"/runs/{body['run_id']}")
        assert response.status_code == 409


def test_sync_invoke_times_out(chain_file):
    slow = MockLLM(delay=0.5)
    with _client(chain_file, slow, chain_timeout_s=0.1) as client:
        body = client.post("/invoke", json={"input": "hi"}).json()
        assert body["status"] == "timeout"
        assert body["success"] is False
        assert "exceeded" in body["error"]
