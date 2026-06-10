"""In-memory chat sessions for deployed agents (PRODUCTION_TODO C3).

Sessions enable a *conversation* with a deployed agent without changing the
chain: the dialogue so far is rendered into the chain's input each turn
(history-in-context). The store keeps the last ``max_turns`` turns per session
and evicts sessions idle past ``ttl_seconds`` (lazy sweep on access). The
clock is injectable so TTL behaviour is testable without sleeping.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel


class Turn(BaseModel):
    user: str
    assistant: str


class Session(BaseModel):
    session_id: str
    turns: list[Turn] = []
    last_active: datetime


def render_history(turns: list[Turn]) -> str:
    """Render prior turns as a plain transcript for history-in-context."""
    lines: list[str] = []
    for turn in turns:
        lines.append(f"User: {turn.user}")
        lines.append(f"Assistant: {turn.assistant}")
    return "\n".join(lines)


def compose_chat_input(turns: list[Turn], message: str) -> str:
    """Build the chain input: prior transcript (if any) + the current message.

    The chain runs on this string as its ``outer_context`` — the chain itself
    is unchanged; it simply sees the conversation so far.
    """
    if not turns:
        return message
    return f"Conversation so far:\n{render_history(turns)}\n\nUser: {message}"


class SessionStore:
    """Per-agent session store: TTL eviction + per-session turn cap."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 1800.0,
        max_turns: int = 6,
        now: Callable[[], datetime] | None = None,
        max_sessions: int = 1000,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._now = now or (lambda: datetime.now(UTC))
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def get_or_create(self, session_id: str | None) -> Session:
        """Return the live session for ``session_id`` (creating a fresh one when
        absent, expired, or ``None``). Expired sessions are swept first."""
        self._evict_expired()
        if session_id and session_id in self._sessions:
            session = self._sessions.pop(session_id)
            session.last_active = self._now()
            self._sessions[session_id] = session  # move to MRU end
            return session
        new_id = session_id or uuid.uuid4().hex[:16]
        session = Session(session_id=new_id, last_active=self._now())
        self._sessions[new_id] = session
        self._evict_overflow()
        return session

    def append_turn(self, session_id: str, user: str, assistant: str) -> Session:
        """Record one completed exchange; trims to the last ``max_turns``."""
        session = self._sessions.get(session_id)
        if session is None:  # expired between the run and the write
            session = Session(session_id=session_id, last_active=self._now())
            self._sessions[session_id] = session
        session.turns.append(Turn(user=user, assistant=assistant))
        if len(session.turns) > self.max_turns:
            session.turns = session.turns[-self.max_turns :]
        session.last_active = self._now()
        return session

    def get(self, session_id: str) -> Session | None:
        self._evict_expired()
        return self._sessions.get(session_id)

    def __len__(self) -> int:
        return len(self._sessions)

    # ----------------------------------------------------------- eviction
    def _evict_expired(self) -> None:
        cutoff = self._now().timestamp() - self.ttl_seconds
        expired = [
            sid
            for sid, s in self._sessions.items()
            if s.last_active.timestamp() < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]

    def _evict_overflow(self) -> None:
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)  # drop the least-recently-used
