"""Agent app surface: health, dynamic Swagger, invoke through real mmar-carl, readiness."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app

from .conftest import SAMPLE_CHAIN, MockLLM


def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def test_healthz(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "agent": "demo"}


def test_info_card(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        info = client.get("/info").json()
        assert info["display_name"] == "Echo Researcher"
        assert info["source"] == "file"
        assert info["ready"] is True
        assert info["version"].startswith("file:")


def test_openapi_is_dynamic_from_chain_meta(chain_file, mock_llm):
    # the whole point of the template: /docs reads as THIS agent's documentation
    with _client(chain_file, mock_llm) as client:
        schema = client.get("/openapi.json").json()
        assert schema["info"]["title"] == "Echo Researcher"
        assert "two reasoning steps" in schema["info"]["description"]
        assert schema["info"]["version"].startswith("file:")
        assert client.get("/docs").status_code == 200


def test_invoke_runs_the_chain_through_real_carl(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        response = client.post("/invoke", json={"input": "what is 6*7?"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "succeeded"
        assert body["success"] is True
        assert body["answer"] == "FINAL ANSWER 42"
        assert len(body["steps"]) == 2
        assert len(mock_llm.calls) == 2  # one LLM call per llm step
        # the run is retrievable afterwards
        run = client.get(f"/runs/{body['run_id']}").json()
        assert run["run_id"] == body["run_id"]
        assert run["status"] == "succeeded"


def test_task_template_wraps_input(chain_file, mock_llm):
    with _client(chain_file, mock_llm, task_template="Context: demo.\nTask: {input}") as client:
        client.post("/invoke", json={"input": "ping"})
        assert "Task: ping" in mock_llm.calls[0]


def test_unknown_run_404(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert client.get("/runs/nope").status_code == 404


def test_invoke_validates_input(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert client.post("/invoke", json={"input": ""}).status_code == 422


def test_readyz_false_on_missing_tool(tmp_path, mock_llm):
    chain = dict(SAMPLE_CHAIN)
    chain["steps"] = [
        {
            "step_type": "tool",
            "number": 1,
            "title": "Call a tool nobody registered",
            "dependencies": [],
            "step_config": {"tool_name": "no_such_tool", "input_mapping": {}},
        }
    ]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(chain), encoding="utf-8")
    with _client(path, mock_llm) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert "no_such_tool" in ready.json()["reason"]
        # invoking a not-ready agent is a clean 503, not a crash
        assert client.post("/invoke", json={"input": "hi"}).status_code == 503


def test_builtin_tools_satisfy_preflight(tmp_path, mock_llm):
    chain = dict(SAMPLE_CHAIN)
    chain["steps"] = [
        {
            "step_type": "tool",
            "number": 1,
            "title": "What time is it",
            "dependencies": [],
            "step_config": {"tool_name": "current_datetime", "input_mapping": {}},
        }
    ]
    path = tmp_path / "tool.json"
    path.write_text(json.dumps(chain), encoding="utf-8")
    with _client(path, mock_llm) as client:
        assert client.get("/readyz").status_code == 200
