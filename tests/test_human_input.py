"""Phase D — human_input pause/resume: a run pauses on a human_input step
(status `waiting`), `POST /runs/{id}/input` resumes it, and the answer reaches
the chain. Driven through the REAL CARL runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app

from .conftest import MockLLM

HUMAN_INPUT_CHAIN = {
    "name": "Greeter",
    "max_workers": 1,
    "timeout": 60.0,
    "steps": [
        {
            "step_type": "human_input",
            "number": 1,
            "title": "Ask name",
            "dependencies": [],
            "step_config": {"prompt": "What is your name?", "output_memory_key": "name"},
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Greet",
            "aim": "Greet the person by the name they gave",
            "reasoning_questions": "",
            "step_context_queries": [],
            "stage_action": "Answer",
            "example_reasoning": "",
            "dependencies": [1],
            "retry_max": 1,
        },
    ],
}


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(HUMAN_INPUT_CHAIN), encoding="utf-8")
    return path


def _client(path: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(path), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def _poll_until(client: TestClient, run_id: str, status: str, deadline_s: float = 10.0) -> dict:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] == status:
            return body
        time.sleep(0.03)
    raise AssertionError(f"run {run_id} never reached {status!r} (last={body['status']})")


def test_pause_then_resume_completes(tmp_path):
    llm = MockLLM(["Hello, Alice!"])
    with _client(_write(tmp_path), llm) as client:
        run_id = client.post("/invoke?mode=async", json={"input": "greet me"}).json()["run_id"]

        waiting = _poll_until(client, run_id, "waiting")
        assert waiting["awaiting_input"]["prompt"] == "What is your name?"

        resumed = client.post(f"/runs/{run_id}/input", json={"value": "Alice"})
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "running"

        done = _poll_until(client, run_id, "succeeded")
        assert done["awaiting_input"] is None  # cleared at terminal
        assert done["answer"] == "Hello, Alice!"
        # the human's answer actually reached the chain (the llm step saw it)
        assert any("Alice" in c for c in llm.calls)


def test_input_on_unknown_run_404(tmp_path, mock_llm):
    with _client(_write(tmp_path), mock_llm) as client:
        assert client.post("/runs/ghost/input", json={"value": "x"}).status_code == 404


def test_input_on_non_waiting_run_409(chain_file, mock_llm):
    # a normal chain (no human_input) finishes without ever waiting
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file))
    with TestClient(build_agent_app(spec, llm_client=mock_llm)) as client:
        run_id = client.post("/invoke", json={"input": "hi"}).json()["run_id"]
        r = client.post(f"/runs/{run_id}/input", json={"value": "x"})
        assert r.status_code == 409
        assert "not waiting" in r.json()["detail"]


def test_snapshot_persisted_on_pause(tmp_path):
    snap_dir = tmp_path / "snaps"
    llm = MockLLM(["Hi!"])
    with _client(_write(tmp_path), llm, snapshot_dir=str(snap_dir)) as client:
        run_id = client.post("/invoke?mode=async", json={"input": "go"}).json()["run_id"]
        _poll_until(client, run_id, "waiting")
        snap_file = snap_dir / f"{run_id}.json"
        assert snap_file.exists()  # durable ContextSnapshot written
        data = json.loads(snap_file.read_text())
        assert "outer_context" in data and "history" in data
        client.post(f"/runs/{run_id}/input", json={"value": "Bob"})
        _poll_until(client, run_id, "succeeded")


def test_input_endpoint_in_openapi(chain_file, mock_llm):
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file))
    with TestClient(build_agent_app(spec, llm_client=mock_llm)) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/runs/{run_id}/input" in paths
