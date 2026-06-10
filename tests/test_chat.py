"""C3 — POST /chat: conversation with history-in-context through the real CARL
runtime. The deciding assertion: the prior turn shows up in the chain's input."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from carl_agent_server import DeploymentSpec, build_agent_app

from .conftest import MockLLM


def _client(chain_file: Path, llm: MockLLM, **spec_kwargs) -> TestClient:
    spec = DeploymentSpec(name="demo", chain_file=str(chain_file), **spec_kwargs)
    return TestClient(build_agent_app(spec, llm_client=llm))


def test_chat_starts_a_session_and_answers(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        response = client.post("/chat", json={"message": "what is 6*7?"})
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"]  # a new id was minted
        assert body["status"] == "succeeded"
        assert body["answer"] == "FINAL ANSWER 42"
        assert body["turn_count"] == 1


def test_chat_feeds_history_into_the_chain():
    """The whole point of C3: turn 2's chain run SEES turn 1."""
    llm = MockLLM(["Paris.", "It has 2.1M people."])  # 1-step chain → 1 call/turn
    spec_file = None

    import json
    import tempfile

    chain = {
        "name": "Geo",
        "max_workers": 1,
        "timeout": 60.0,
        "steps": [
            {
                "step_type": "llm",
                "number": 1,
                "title": "Answer",
                "aim": "Answer the question",
                "reasoning_questions": "",
                "step_context_queries": [],
                "stage_action": "Answer",
                "example_reasoning": "",
                "dependencies": [],
                "retry_max": 1,
            }
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(chain, fh)
        spec_file = fh.name

    with _client(Path(spec_file), llm) as client:
        first = client.post("/chat", json={"message": "capital of France?"}).json()
        session_id = first["session_id"]
        assert first["answer"] == "Paris."

        second = client.post(
            "/chat", json={"message": "how many people?", "session_id": session_id}
        ).json()
        assert second["session_id"] == session_id
        assert second["turn_count"] == 2
        # the SECOND run's prompt carried the first exchange (history-in-context)
        last_prompt = llm.calls[-1]
        assert "Conversation so far:" in last_prompt
        assert "capital of France?" in last_prompt
        assert "Assistant: Paris." in last_prompt
        assert "User: how many people?" in last_prompt


def test_chat_unknown_session_starts_fresh(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        body = client.post(
            "/chat", json={"message": "hi", "session_id": "does-not-exist"}
        ).json()
        # an explicit id is honoured even when new; turn 1 has no history
        assert body["session_id"] == "does-not-exist"
        assert body["turn_count"] == 1


def test_chat_validates_message(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_503_when_not_ready(tmp_path, mock_llm):
    import json

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "name": "x",
                "steps": [
                    {
                        "step_type": "tool",
                        "number": 1,
                        "title": "T",
                        "dependencies": [],
                        "step_config": {"tool_name": "no_such_tool", "input_mapping": {}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with _client(bad, mock_llm) as client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 503


def test_chat_in_openapi(chain_file, mock_llm):
    with _client(chain_file, mock_llm) as client:
        schema = client.get("/openapi.json").json()
        assert "/chat" in schema["paths"]
