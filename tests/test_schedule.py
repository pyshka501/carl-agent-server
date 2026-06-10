"""D3 — scheduled auto-invocation: a deployment fires its chain on a cadence,
plus a manual trigger. The in-template scheduler (external cron = `care run`)."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, ScheduleConfig, build_agent_app

from .conftest import MockLLM


def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def test_no_schedule_by_default(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        body = client.get("/schedule").json()
        assert body["configured"] is False
        assert body["fire_count"] == 0


def test_schedule_reported(chain_file, mock_llm):
    sched = ScheduleConfig(interval_s=30.0, input="tick")
    with _client(chain_file, mock_llm, schedule=sched) as client:
        body = client.get("/schedule").json()
        assert body["configured"] is True
        assert body["enabled"] is True
        assert body["interval_s"] == 30.0
        assert body["input"] == "tick"


def test_scheduler_fires_on_cadence(chain_file, mock_llm):
    sched = ScheduleConfig(interval_s=0.05, input="auto run")
    with _client(chain_file, mock_llm, schedule=sched) as client:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if client.get("/schedule").json()["fire_count"] >= 2:
                break
            time.sleep(0.05)
        status = client.get("/schedule").json()
        assert status["fire_count"] >= 2  # the timer fired repeatedly
        assert status["last_run_id"]
        # the auto-runs are real runs with the scheduled input
        runs = client.get(f"/runs/{status['last_run_id']}").json()
        assert runs["input"] == "auto run"


def test_disabled_schedule_does_not_fire(chain_file, mock_llm):
    sched = ScheduleConfig(interval_s=0.05, input="tick", enabled=False)
    with _client(chain_file, mock_llm, schedule=sched) as client:
        time.sleep(0.3)
        status = client.get("/schedule").json()
        assert status["configured"] is True
        assert status["enabled"] is False
        assert status["fire_count"] == 0  # paused — never fires


def test_manual_trigger(chain_file, mock_llm):
    sched = ScheduleConfig(interval_s=3600.0, input="manual")  # long interval: no auto-fire
    with _client(chain_file, mock_llm, schedule=sched) as client:
        response = client.post("/schedule/trigger")
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert client.get("/schedule").json()["fire_count"] == 1
        # the triggered run completes
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            body = client.get(f"/runs/{run_id}").json()
            if body["status"] != "running":
                break
            time.sleep(0.03)
        assert body["status"] == "succeeded"
        assert body["input"] == "manual"


def test_trigger_404_without_schedule(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert client.post("/schedule/trigger").status_code == 404


def test_trigger_requires_key_when_set(chain_file, mock_llm):
    sched = ScheduleConfig(interval_s=3600.0, input="x")
    spec = DeploymentSpec(
        name="demo", chain_file=str(chain_file), schedule=sched,
        api_key="k", auth_allow_localhost=False,
    )
    with TestClient(build_agent_app(spec, llm_client=mock_llm)) as client:
        assert client.post("/schedule/trigger").status_code == 401
        assert client.get("/schedule").status_code == 200  # read stays open


def test_schedule_in_openapi(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/schedule" in paths
        assert "/schedule/trigger" in paths
