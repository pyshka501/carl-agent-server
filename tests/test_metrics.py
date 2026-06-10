"""D4 — USD cost tracking, per-deployment budget (402), and the /metrics report."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app
from carl_agent_server.agent import AgentState
from carl_agent_server.cost import run_cost_usd
from carl_agent_server.models import RunRecord

from .conftest import MockLLM


# ------------------------------------------------------------- cost unit
def test_cost_priced():
    assert run_cost_usd({"prompt": 1000, "completion": 500}, 0.001, 0.002) == 0.002


def test_cost_no_pricing_is_none():
    assert run_cost_usd({"prompt": 1000}, None, 0.002) is None
    assert run_cost_usd({"prompt": 1000}, 0.001, None) is None


def test_cost_key_variants_and_empty():
    assert run_cost_usd({"prompt_tokens": 2000, "completion_tokens": 0}, 0.001, 0.002) == 0.002
    assert run_cost_usd({}, 0.001, 0.002) == 0.0
    assert run_cost_usd(None, 0.001, 0.002) == 0.0


# ------------------------------------------------------ metrics fold
def _state(tmp_path: Path, **spec_kwargs) -> AgentState:
    chain = tmp_path / "c.json"
    chain.write_text('{"name":"c","steps":[]}', encoding="utf-8")
    return AgentState(
        DeploymentSpec(name="d", chain_file=str(chain), **spec_kwargs), llm_client=MockLLM()
    )


def _rec(status: str, prompt: int, completion: int) -> RunRecord:
    return RunRecord(
        input="x", status=status,
        token_usage={"prompt": prompt, "completion": completion, "total": prompt + completion},
    )


def test_metrics_fold_accumulates_cost(tmp_path):
    state = _state(tmp_path, price_per_1k_input_usd=0.001, price_per_1k_output_usd=0.002)
    r1 = _rec("succeeded", 1000, 500)  # $0.002
    r2 = _rec("failed", 1000, 0)       # $0.001
    state._record_metrics(r1)
    state._record_metrics(r2)
    assert r1.cost_usd == 0.002  # stamped on the record
    report = state.metrics_report()
    assert report.run_count == 2
    assert report.status_counts == {"succeeded": 1, "failed": 1}
    assert report.total_tokens == 1500 + 1000
    assert abs(report.total_cost_usd - 0.003) < 1e-9
    assert report.pricing_configured is True


def test_metrics_without_pricing(tmp_path):
    state = _state(tmp_path)  # no prices
    state._record_metrics(_rec("succeeded", 1000, 500))
    report = state.metrics_report()
    assert report.run_count == 1
    assert report.pricing_configured is False
    assert report.total_cost_usd is None
    assert report.total_tokens == 1500


def test_budget_remaining_and_over(tmp_path):
    state = _state(
        tmp_path, price_per_1k_input_usd=0.001, price_per_1k_output_usd=0.002, budget_usd=0.005
    )
    state._record_metrics(_rec("succeeded", 1000, 500))  # $0.002
    report = state.metrics_report()
    assert abs(report.remaining_usd - 0.003) < 1e-9
    assert report.over_budget is False
    # push past the budget
    state._record_metrics(_rec("succeeded", 2000, 500))  # +$0.003 -> $0.005 == budget
    assert state.over_budget() is True
    assert state.metrics_report().remaining_usd == 0.0


def test_budget_needs_pricing(tmp_path):
    state = _state(tmp_path, budget_usd=0.001)  # budget but NO pricing
    state._metrics_cost_usd = 999.0  # irrelevant without pricing
    assert state.over_budget() is False  # can't enforce a budget we can't price


# ----------------------------------------------------- endpoints
def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def test_metrics_endpoint_grows_with_runs(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert client.get("/metrics").json()["run_count"] == 0
        client.post("/invoke", json={"input": "hi"})
        body = client.get("/metrics").json()
        assert body["run_count"] == 1
        assert body["status_counts"]["succeeded"] == 1


def test_invoke_402_when_over_budget(chain_file, mock_llm):
    with _client(
        chain_file, mock_llm,
        price_per_1k_input_usd=0.001, price_per_1k_output_usd=0.002, budget_usd=0.5,
    ) as client:
        state = client.app.state.agent
        state._metrics_cost_usd = 0.5  # at the cap
        r = client.post("/invoke", json={"input": "hi"})
        assert r.status_code == 402
        assert "budget" in r.json()["detail"]
        # /metrics still readable and flags over_budget
        assert client.get("/metrics").json()["over_budget"] is True


def test_metrics_in_openapi(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert "/metrics" in client.get("/openapi.json").json()["paths"]
