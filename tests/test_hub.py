"""A4 — the hub: control API, dynamic mount/unmount, per-agent /docs, state restore."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app, build_hub_app

from .conftest import SAMPLE_CHAIN, MockLLM


def _write_chain(path: Path, name: str) -> Path:
    chain = dict(SAMPLE_CHAIN, name=name)
    path.write_text(json.dumps(chain), encoding="utf-8")
    return path


def _factory(spec: DeploymentSpec):
    return build_agent_app(spec, llm_client=MockLLM())


def _hub(tmp_path: Path, state_file: Path | None = None) -> TestClient:
    return TestClient(build_hub_app(state_file=state_file, agent_app_factory=_factory))


def test_two_agents_each_with_own_docs(tmp_path):
    alpha = _write_chain(tmp_path / "a.json", "Alpha Chain")
    beta = _write_chain(tmp_path / "b.json", "Beta Chain")
    with _hub(tmp_path) as client:
        for name, path in (("alpha", alpha), ("beta", beta)):
            response = client.post("/deployments", json={"name": name, "chain_file": str(path)})
            assert response.status_code == 201, response.text
            assert response.json()["url"] == f"/agents/{name}"
        # the deciding assertion of the design: EACH mounted agent serves its OWN Swagger
        for name, title in (("alpha", "Alpha Chain"), ("beta", "Beta Chain")):
            assert client.get(f"/agents/{name}/docs").status_code == 200
            schema = client.get(f"/agents/{name}/openapi.json").json()
            assert schema["info"]["title"] == title
        # and each answers independently
        body = client.post("/agents/alpha/invoke", json={"input": "hi"}).json()
        assert body["status"] == "succeeded"
        listing = client.get("/deployments").json()
        assert {d["name"] for d in listing} == {"alpha", "beta"}
        assert all(d["ready"] for d in listing)
        assert client.get("/healthz").json() == {"status": "ok", "deployments": 2}


def test_duplicate_name_conflicts(tmp_path):
    chain = _write_chain(tmp_path / "a.json", "Alpha Chain")
    with _hub(tmp_path) as client:
        assert client.post("/deployments", json={"name": "x", "chain_file": str(chain)}).status_code == 201
        assert client.post("/deployments", json={"name": "x", "chain_file": str(chain)}).status_code == 409


def test_undeploy_unmounts_the_agent(tmp_path):
    chain = _write_chain(tmp_path / "a.json", "Alpha Chain")
    with _hub(tmp_path) as client:
        client.post("/deployments", json={"name": "x", "chain_file": str(chain)})
        assert client.get("/agents/x/healthz").status_code == 200
        assert client.delete("/deployments/x").status_code == 200
        assert client.get("/agents/x/healthz").status_code == 404  # mount removed
        assert client.get("/deployments").json() == []
        assert client.delete("/deployments/x").status_code == 404


def test_reload_picks_up_source_changes(tmp_path):
    chain = _write_chain(tmp_path / "a.json", "Alpha Chain")
    with _hub(tmp_path) as client:
        client.post("/deployments", json={"name": "x", "chain_file": str(chain)})
        _write_chain(chain, "Alpha Chain 2.0")  # edit the source in place
        response = client.post("/deployments/x/reload")
        assert response.status_code == 200
        assert response.json()["reloaded"] is True
        assert response.json()["deployment"]["display_name"] == "Alpha Chain 2.0"
        # the mounted agent's own docs follow the reload
        assert client.get("/agents/x/openapi.json").json()["info"]["title"] == "Alpha Chain 2.0"


def test_deploy_unloadable_chain_is_rejected(tmp_path):
    with _hub(tmp_path) as client:
        response = client.post("/deployments", json={"name": "x", "chain_file": str(tmp_path / "missing.json")})
        assert response.status_code == 422
        assert client.get("/deployments").json() == []  # nothing mounted, nothing persisted


def test_state_persists_and_restores(tmp_path):
    chain = _write_chain(tmp_path / "a.json", "Alpha Chain")
    state_file = tmp_path / "hub-state.json"
    with _hub(tmp_path, state_file=state_file) as client:
        client.post("/deployments", json={"name": "alpha", "chain_file": str(chain)})
    saved = json.loads(state_file.read_text())
    assert saved["deployments"][0]["name"] == "alpha"
    # a NEW hub process restores the deployment from the state file
    with _hub(tmp_path, state_file=state_file) as client:
        listing = client.get("/deployments").json()
        assert [d["name"] for d in listing] == ["alpha"]
        assert client.post("/agents/alpha/invoke", json={"input": "hi"}).json()["status"] == "succeeded"
        # undeploy persists too
        client.delete("/deployments/alpha")
    assert json.loads(state_file.read_text())["deployments"] == []


def test_unknown_deployment_404(tmp_path):
    with _hub(tmp_path) as client:
        assert client.get("/deployments/ghost").status_code == 404
        assert client.post("/deployments/ghost/reload").status_code == 404
