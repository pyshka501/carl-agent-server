"""C3 — SessionStore: history-in-context, turn cap, TTL eviction, LRU overflow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from carl_agent_server.sessions import (
    SessionStore,
    Turn,
    compose_chat_input,
    render_history,
)


class _Clock:
    def __init__(self) -> None:
        self.t = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t = self.t + timedelta(seconds=seconds)


def test_compose_first_message_is_bare():
    assert compose_chat_input([], "hi") == "hi"


def test_compose_includes_history():
    turns = [Turn(user="what is 2+2?", assistant="4")]
    composed = compose_chat_input(turns, "and times 3?")
    assert "Conversation so far:" in composed
    assert "User: what is 2+2?" in composed
    assert "Assistant: 4" in composed
    assert composed.endswith("User: and times 3?")


def test_render_history():
    out = render_history([Turn(user="a", assistant="b"), Turn(user="c", assistant="d")])
    assert out == "User: a\nAssistant: b\nUser: c\nAssistant: d"


def test_get_or_create_new_session_gets_id():
    store = SessionStore()
    session = store.get_or_create(None)
    assert session.session_id
    assert session.turns == []
    # round-trips by id
    again = store.get_or_create(session.session_id)
    assert again.session_id == session.session_id


def test_append_and_turn_cap():
    store = SessionStore(max_turns=2)
    s = store.get_or_create("s1")
    store.append_turn("s1", "q1", "a1")
    store.append_turn("s1", "q2", "a2")
    store.append_turn("s1", "q3", "a3")
    s = store.get("s1")
    assert [t.user for t in s.turns] == ["q2", "q3"]  # capped to the last 2


def test_ttl_eviction():
    clock = _Clock()
    store = SessionStore(ttl_seconds=60, now=clock)
    store.get_or_create("s1")
    store.append_turn("s1", "q", "a")
    clock.advance(61)
    # the expired session is gone; a same-id request starts fresh
    assert store.get("s1") is None
    s = store.get_or_create("s1")
    assert s.turns == []


def test_activity_refreshes_ttl():
    clock = _Clock()
    store = SessionStore(ttl_seconds=60, now=clock)
    store.get_or_create("s1")
    clock.advance(40)
    store.get_or_create("s1")  # touch -> last_active refreshed
    clock.advance(40)  # 80s since creation but only 40s since touch
    assert store.get("s1") is not None


def test_lru_overflow_eviction():
    store = SessionStore(max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("c")  # evicts the LRU ("a")
    assert store.get("a") is None
    assert store.get("b") is not None
    assert store.get("c") is not None
    assert len(store) == 2
