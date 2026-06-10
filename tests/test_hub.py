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


def test_attached_agent_hot_reloads_inside_hub():
    """B5 e2e: deploy(attached) → promote on the channel → the MOUNTED agent
    hot-reloads (its /docs + the hub listing show the new version) and keeps
    answering. The full release path under the hub, no network."""
    import time

    from carl_agent_server.chain_source import MemoryChainSource

    from .test_attached import FakeMemoryClient, _record

    fake = FakeMemoryClient(_record(1, SAMPLE_CHAIN))

    def factory(spec: DeploymentSpec):
        source = MemoryChainSource("chain-1", channel="stable", client=fake)
        return build_agent_app(spec, chain_source=source, llm_client=MockLLM())

    app = build_hub_app(state_file=None, agent_app_factory=factory)
    with TestClient(app) as client:
        created = client.post(
            "/deployments", json={"name": "weather", "entity_id": "chain-1"}
        )
        assert created.status_code == 201
        assert created.json()["version"].startswith("v1")
        assert fake.callback is not None  # the mounted agent armed its watcher

        fake.promote(_record(2, dict(SAMPLE_CHAIN), display_name="Weather 2.0"))

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            info = client.get("/agents/weather/info").json()
            if info["version"].startswith("v2"):
                break
            time.sleep(0.05)
        else:  # pragma: no cover
            raise AssertionError(f"agent never reloaded: {info}")
        # the mounted agent's own Swagger follows the release
        assert client.get("/agents/weather/openapi.json").json()["info"]["title"] == "Weather 2.0"
        # the hub listing reflects the live version too
        listing = client.get("/deployments").json()
        assert listing[0]["version"].startswith("v2")
        # and the rolled-out agent still answers
        assert client.post("/agents/weather/invoke", json={"input": "hi"}).json()["status"] == "succeeded"
    assert fake.subscription.stopped is True  # hub shutdown stopped the watcher
