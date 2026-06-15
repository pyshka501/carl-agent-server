"""Chain sources — where an agent's chain comes from (a file or gigaevo Memory)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import AgentMeta


@dataclass
class ChainSnapshot:
    """A loaded chain: raw CARL JSON content + presentation metadata."""

    content: dict[str, Any]
    meta: AgentMeta


class ChainSource(Protocol):
    def load(self) -> ChainSnapshot:
        """Fetch the current chain. Called in a thread — may block."""
        ...


class FileChainSource:
    """Offline source: chain JSON from disk (solo mode / tests / exported agents)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> ChainSnapshot:
        content = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(content, dict):
            raise ValueError(f"{self.path}: chain JSON must be an object")
        meta = AgentMeta(
            display_name=str(content.get("name") or self.path.stem),
            description=str(content.get("description") or ""),
            version_label=f"file:{self.path.name}",
            source="file",
        )
        return ChainSnapshot(content=content, meta=meta)


class MemoryChainSource:
    """Attached source: a chain entity in gigaevo Memory, following a channel.

    Presentation metadata (display_name / description / when_to_use / the
    example task) comes from the ENTITY record, the executable chain from the
    resolved version's content.
    """

    def __init__(
        self,
        entity_id: str,
        *,
        channel: str = "stable",
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not entity_id:
            raise ValueError("entity_id is required for MemoryChainSource")
        self.entity_id = entity_id
        self.channel = channel
        self._base_url = base_url
        self._api_key = api_key
        self._client = client

    def get_client(self) -> Any:
        """The (lazily constructed) Memory client — shared by load/watch/run-records."""
        if self._client is None:
            from gigaevo_client import MemoryClient

            kwargs: dict[str, Any] = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = MemoryClient(**kwargs)
        return self._client

    def watch(self, callback: Any) -> Any:
        """Subscribe to entity events on the followed channel (SSE, background thread).

        Returns the started ``gigaevo_client`` Subscription (``.stop()`` to end).
        The callback receives a fresh ``ChainRecord`` per event; the agent
        ignores the payload and re-loads through :meth:`load` so reload and
        hot-reload share one code path.
        """
        return self.get_client().watch_chain_record(self.entity_id, callback, channel=self.channel)

    def head(self) -> str:
        """Identity (version_id) of the followed channel's current version.

        The poll-fallback probe: compared against the serving version to detect
        promotes the SSE watcher missed. Called in a thread — may block.
        """
        record = self.get_client().get_chain_record(self.entity_id, channel=self.channel)
        return str(getattr(record, "version_id", "") or "")

    def load(self) -> ChainSnapshot:
        record = self.get_client().get_chain_record(self.entity_id, channel=self.channel)
        record_meta: dict[str, Any] = dict(getattr(record, "meta", None) or {})
        content: dict[str, Any] = dict(getattr(record, "content", None) or {})
        version_number = getattr(record, "version_number", None)
        version_id = str(getattr(record, "version_id", "") or "")
        if version_number is not None:
            version_label = f"v{version_number} ({version_id[:8]})" if version_id else f"v{version_number}"
        else:
            version_label = version_id[:8] or "unknown"
        meta = AgentMeta(
            display_name=str(
                record_meta.get("display_name") or record_meta.get("name") or content.get("name") or self.entity_id
            ),
            description=str(record_meta.get("description") or content.get("description") or ""),
            when_to_use=str(record_meta.get("when_to_use") or ""),
            example_task=str(record_meta.get("task_description") or record_meta.get("query") or ""),
            version_label=version_label,
            version_id=version_id,
            entity_id=self.entity_id,
            channel=self.channel,
            source="memory",
        )
        return ChainSnapshot(content=content, meta=meta)
