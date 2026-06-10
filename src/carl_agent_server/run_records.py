"""Best-effort run-records: every finished run lands in gigaevo Memory.

The card mirrors CARE's ``run_recorder`` shape (category ``agent_run``,
``usage.run_id``/metrics, tags ``agent_run`` / ``agent:<entity>`` /
``status:<label>``), so hub runs and TUI runs read through ONE replay/history
mechanism. On top of the card the chain entity's ``run_count``/``last_run_at``
are bumped via ``record_chain_run``.

Strictly best-effort: any Memory failure logs a warning and NEVER fails the
invoke — answering the user always outranks bookkeeping.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .models import RunRecord

logger = logging.getLogger(__name__)

_STATUS_LABELS = {"succeeded": "success"}  # CARE uses success/failed; the rest pass through


def status_label(record: RunRecord) -> str:
    return _STATUS_LABELS.get(record.status, record.status)


class RunRecorder:
    """Writes one memory_card + one run-recorded ping per finished run."""

    def __init__(self, client: Any, *, entity_id: str, agent_name: str) -> None:
        self.client = client
        self.entity_id = entity_id
        self.agent_name = agent_name

    async def record(self, record: RunRecord) -> None:
        try:
            await asyncio.to_thread(self._record_sync, record)
        except Exception as exc:
            logger.warning("agent %s: run-record failed for %s (%s)", self.agent_name, record.run_id, exc)

    def _record_sync(self, record: RunRecord) -> None:
        label = status_label(record)
        duration = record.execution_time_s
        if duration is None and record.finished_at is not None:
            duration = (record.finished_at - record.created_at).total_seconds()
        metrics: dict[str, Any] = {
            "duration_seconds": duration or 0.0,
            "step_count": len(record.steps),
            "exit_status": label,
        }
        total_tokens = record.token_usage.get("total")
        if total_tokens is not None:
            metrics["total_tokens"] = total_tokens

        description = f"Run of agent {self.agent_name} — {label}"
        if record.error:
            description = f"{description}: {record.error[:200]}"

        content = {
            "category": "agent_run",
            "task_description": record.input,
            "description": description,
            "keywords": ["agent_run", f"agent:{self.entity_id}", label],
            "usage": {
                "run_id": record.run_id,
                "agent_entity_id": self.entity_id,
                "agent_name": self.agent_name,
                "finished_at": record.finished_at.isoformat() if record.finished_at else None,
                "metrics": metrics,
                "answer_preview": (record.answer or "")[:500],
            },
        }
        self.client.save_memory_card(
            content,
            name=f"{self.agent_name} · {label} · {record.run_id}",
            tags=[
                "agent_run",
                f"agent:{self.entity_id}",
                f"status:{label}",
                "source:agent-server",
                f"deployment:{self.agent_name}",
            ],
            when_to_use=f"Replay context / debug for run {record.run_id} of agent {self.agent_name}.",
        )
        self.client.record_chain_run(self.entity_id, run_id=record.run_id)
