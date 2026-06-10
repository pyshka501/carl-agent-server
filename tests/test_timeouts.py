"""C6/G9 — default-timeout injection: fill gaps, never loosen the author's."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app
from carl_agent_server.timeouts import inject_default_timeouts

from .conftest import MockLLM


def _chain(*, chain_timeout=None, step_timeouts=None) -> dict:
    steps = []
    for i, t in enumerate(step_timeouts or [None, None], start=1):
        step = {"step_type": "llm", "number": i, "title": f"S{i}", "dependencies": []}
        if t is not None:
            step["timeout"] = t
        steps.append(step)
    chain: dict = {"name": "C", "steps": steps}
    if chain_timeout is not None:
        chain["timeout"] = chain_timeout
    return chain


def test_injects_chain_and_step_defaults_when_absent():
    out = inject_default_timeouts(_chain(), step_timeout_s=60.0)
    assert out["timeout"] == 60.0  # chain-level default filled
    assert [s["timeout"] for s in out["steps"]] == [60.0, 60.0]


def test_preserves_authored_step_timeout():
    out = inject_default_timeouts(
        _chain(chain_timeout=120, step_timeouts=[10, None]), step_timeout_s=60.0
    )
    assert out["steps"][0]["timeout"] == 10  # authored kept verbatim
    # the gap step is capped at the per-step default (< chain 120)
    assert out["steps"][1]["timeout"] == 60.0


def test_never_loosens_authored_chain_timeout():
    # author meant "each step <= 30s"; the 60s default must NOT loosen that
    out = inject_default_timeouts(
        _chain(chain_timeout=30), step_timeout_s=60.0
    )
    assert out["timeout"] == 30  # preserved
    assert [s["timeout"] for s in out["steps"]] == [30.0, 30.0]  # min(60, 30)


def test_caps_steps_below_a_large_chain_timeout():
    out = inject_default_timeouts(
        _chain(chain_timeout=300), step_timeout_s=60.0
    )
    assert out["timeout"] == 300  # the overall chain budget is preserved
    assert [s["timeout"] for s in out["steps"]] == [60.0, 60.0]  # each step capped


def test_pure_does_not_mutate_input():
    original = _chain()
    inject_default_timeouts(original, step_timeout_s=60.0)
    assert "timeout" not in original
    assert all("timeout" not in s for s in original["steps"])


def test_ignores_nonpositive_authored_values():
    out = inject_default_timeouts(
        _chain(chain_timeout=0, step_timeouts=[-5, None]), step_timeout_s=60.0
    )
    assert out["timeout"] == 60.0  # 0 treated as absent
    assert out["steps"][0]["timeout"] == 60.0  # -5 treated as absent


# ----------------------------------------------------------- integration
def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def test_loaded_chain_carries_injected_timeouts(tmp_path, mock_llm):
    import json

    # a chain whose steps have NO per-step timeout
    chain = _chain(step_timeouts=[None, None])
    for step in chain["steps"]:
        step.update(
            aim="A", reasoning_questions="", step_context_queries=[],
            stage_action="Answer", example_reasoning="", retry_max=1,
        )
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(chain), encoding="utf-8")

    with _client(path, mock_llm, step_timeout_s=45.0) as client:
        assert client.get("/readyz").status_code == 200
        state = client.app.state.agent
        loaded = state.chain.to_dict()
        assert loaded["timeout"] == 45.0
        assert all(s["timeout"] == 45.0 for s in loaded["steps"])
