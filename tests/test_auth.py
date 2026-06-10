"""C4 — per-agent API key: protected routes require it, open routes don't,
loopback bypass is configurable. TestClient's host is "testclient" (not
loopback), so the key IS enforced unless we override the client host."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app

from .conftest import MockLLM

KEY = "sek-ret-123"


def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def test_no_key_means_open(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:  # api_key=None
        assert client.post("/invoke", json={"input": "hi"}).status_code == 200


def test_protected_routes_require_key(chain_file, mock_llm):
    with _client(chain_file, mock_llm, api_key=KEY, auth_allow_localhost=False) as client:
        assert client.post("/invoke", json={"input": "hi"}).status_code == 401
        assert client.post("/chat", json={"message": "hi"}).status_code == 401
        assert client.get("/runs/whatever").status_code == 401
        assert client.delete("/runs/whatever").status_code == 401


def test_open_routes_never_need_key(chain_file, mock_llm):
    with _client(chain_file, mock_llm, api_key=KEY, auth_allow_localhost=False) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/info").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_valid_key_via_x_api_key(chain_file, mock_llm):
    with _client(chain_file, mock_llm, api_key=KEY, auth_allow_localhost=False) as client:
        ok = client.post("/invoke", json={"input": "hi"}, headers={"X-API-Key": KEY})
        assert ok.status_code == 200
        bad = client.post("/invoke", json={"input": "hi"}, headers={"X-API-Key": "wrong"})
        assert bad.status_code == 401


def test_valid_key_via_bearer(chain_file, mock_llm):
    with _client(chain_file, mock_llm, api_key=KEY, auth_allow_localhost=False) as client:
        ok = client.post(
            "/invoke", json={"input": "hi"}, headers={"Authorization": f"Bearer {KEY}"}
        )
        assert ok.status_code == 200


def test_localhost_bypass(chain_file, mock_llm):
    # auth_allow_localhost=True (default) + a loopback client host -> no key needed
    with _client(chain_file, mock_llm, api_key=KEY) as client:
        response = client.post(
            "/invoke", json={"input": "hi"}, headers={"X-API-Key": "wrong"}
        )
        # TestClient default host is "testclient"; simulate loopback via the
        # forwarded client tuple ASGI scope uses for request.client
        assert response.status_code == 401  # not loopback -> still enforced

    # now prove the loopback path: a request whose client host is 127.0.0.1
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), api_key=KEY)
    app = build_agent_app(spec, llm_client=MockLLM())
    with TestClient(app, client=("127.0.0.1", 50000)) as loopback_client:
        ok = loopback_client.post("/invoke", json={"input": "hi"})  # no key
        assert ok.status_code == 200


def test_localhost_bypass_disabled(chain_file, mock_llm):
    spec = DeploymentSpec(
        name="demo", chain_file=str(chain_file), api_key=KEY, auth_allow_localhost=False
    )
    app = build_agent_app(spec, llm_client=MockLLM())
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.post("/invoke", json={"input": "hi"}).status_code == 401
