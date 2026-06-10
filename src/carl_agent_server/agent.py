"""AgentState — one chain behind an HTTP facade: load, preflight, runs (sync/async/SSE/cancel)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chain_source import ChainSnapshot, ChainSource, FileChainSource, MemoryChainSource
from .llm import LLMNotConfiguredError, build_llm_client
from .models import AgentMeta, DeploymentSpec, RunRecord, ScheduleStatus, StepSummary
from .run_records import RunRecorder
from .sessions import SessionStore, compose_chat_input
from .timeouts import inject_default_timeouts
from .tools import register_builtin_tools

logger = logging.getLogger(__name__)

_MAX_RUNS_KEPT = 200
ENV_WEB_SEARCH_KEY = "AGENT_WEB_SEARCH_API_KEY"


class _NullLLM:
    """Placeholder api object for preflight-only contexts. Never actually called."""

    async def get_response(self, *args: Any, **kwargs: Any) -> str:
        raise LLMNotConfiguredError("LLM is not configured")


class RunHandle:
    """A live run: its record, step-event log for SSE, and the cancel lever.

    Events are kept for the run's lifetime, so a late SSE subscriber replays
    the full step history before tailing new events.
    """

    def __init__(self, record: RunRecord) -> None:
        self.record = record
        self.context: Any | None = None
        self.task: asyncio.Task[None] | None = None
        self.cancel_requested = False
        self.awaiting_input = False
        self.input_future: Any | None = None
        self.snapshot: Any | None = None
        self.events: list[dict[str, Any]] = []
        self.finished = False
        self._cond = asyncio.Condition()

    async def emit(self, event: str, data: dict[str, Any]) -> None:
        async with self._cond:
            self.events.append({"event": event, "data": data})
            if event == "result":
                self.finished = True
            self._cond.notify_all()

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Replay existing events, then tail until the terminal ``result`` event."""
        index = 0
        while True:
            async with self._cond:
                while index >= len(self.events) and not self.finished:
                    await self._cond.wait()
                if index >= len(self.events):
                    return
                event = self.events[index]
                index += 1
            yield event
            if event["event"] == "result":
                return

    def request_cancel(self) -> None:
        """Cooperative cancel: CARL's executor stops between batches."""
        self.cancel_requested = True
        if self.context is not None:
            self.context.cancel()


class AgentState:
    """Holds the loaded chain + metadata and executes runs.

    ``reload()`` is swap-safe: a failed re-load (fetch / parse / preflight)
    keeps the previously working chain — the canary behaviour the hot-reload
    path relies on.
    """

    def __init__(
        self,
        spec: DeploymentSpec,
        *,
        chain_source: ChainSource | None = None,
        llm_client: Any | None = None,
        run_recorder: RunRecorder | None = None,
    ) -> None:
        self.spec = spec
        self._source: ChainSource = chain_source or self._default_source(spec)
        self._llm = llm_client
        self._recorder = run_recorder
        self._recorder_resolved = run_recorder is not None
        self.chain: Any | None = None
        self.meta: AgentMeta = AgentMeta(display_name=spec.name)
        self.required_tools: list[str] = []
        self.missing: list[str] = []
        self.load_error: str | None = None
        self.loaded_at: datetime | None = None
        self.runs: OrderedDict[str, RunRecord] = OrderedDict()
        self.handles: dict[str, RunHandle] = {}
        self.sessions = SessionStore(
            ttl_seconds=spec.session_ttl_s, max_turns=spec.session_history_turns
        )
        self.on_reloaded: list[Callable[[], None]] = []
        self._swap_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        # D3 — auto-invoke scheduler
        self._scheduler_task: asyncio.Task[None] | None = None
        self._schedule_fire_count = 0
        self._schedule_last_fired: datetime | None = None
        self._schedule_last_run_id: str | None = None
        self._loaded_once = False
        self._subscription: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------- lifecycle
    @staticmethod
    def _default_source(spec: DeploymentSpec) -> ChainSource:
        if spec.chain_file:
            return FileChainSource(spec.chain_file)
        return MemoryChainSource(
            spec.entity_id or "",
            channel=spec.channel,
            base_url=spec.memory_url or os.environ.get("AGENT_MEMORY_URL"),
            api_key=spec.memory_api_key or os.environ.get("AGENT_MEMORY_API_KEY"),
        )

    async def ensure_loaded(self) -> None:
        if not self._loaded_once:
            await self.reload()

    async def reload(self) -> bool:
        """(Re)load the chain from its source. Returns True if the new chain went live.

        Serialized (concurrent watcher events queue up) and swap-safe: on any
        failure the previous chain (if any) stays serving.
        """
        async with self._reload_lock:
            went_live = await self._reload_inner()
        if went_live:
            for hook in list(self.on_reloaded):
                try:
                    hook()
                except Exception:
                    logger.exception("agent %s: on_reloaded hook failed", self.spec.name)
        return went_live

    async def _reload_inner(self) -> bool:
        self._loaded_once = True
        try:
            snapshot = await asyncio.to_thread(self._source.load)
            chain, required, missing = self._parse_and_preflight(snapshot)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("agent %s: reload failed, keeping previous chain (%s)", self.spec.name, self.load_error)
            return False
        if missing and self.chain is not None:
            # Preflight canary: a hot-reloaded version with unmet dependencies must
            # not evict a healthy serving chain. (On FIRST load we do accept it, so
            # /readyz can report exactly what is missing.)
            logger.warning(
                "agent %s: rejecting %s — preflight missing %s; keeping %s",
                self.spec.name,
                snapshot.meta.version_label,
                ", ".join(missing),
                self.meta.version_label,
            )
            return False
        async with self._swap_lock:
            self.chain = chain
            self.meta = snapshot.meta
            self.required_tools = required
            self.missing = missing
            self.load_error = None
            self.loaded_at = datetime.now(UTC)
        logger.info("agent %s: serving %s (%s)", self.spec.name, self.meta.display_name, self.meta.version_label)
        return True

    # ------------------------------------------------------------ hot-reload
    def start_watch(self) -> bool:
        """Subscribe to source events (attached mode) for promote/pin hot-reload.

        Returns True if a watcher is now active. File sources have nothing to
        watch; calling twice is a no-op. Must be called from a running loop —
        watcher events arrive on a background thread and are bridged back via
        ``run_coroutine_threadsafe``.
        """
        watch = getattr(self._source, "watch", None)
        if watch is None or self._subscription is not None:
            return False
        self._loop = asyncio.get_running_loop()
        try:
            self._subscription = watch(self._on_source_event)
        except Exception as exc:
            logger.warning("agent %s: could not start watcher (%s); serving without hot-reload", self.spec.name, exc)
            return False
        logger.info("agent %s: watching channel %r for updates", self.spec.name, self.spec.channel)
        return True

    def stop_watch(self) -> None:
        subscription, self._subscription = self._subscription, None
        if subscription is not None:
            try:
                subscription.stop()
            except Exception:
                logger.exception("agent %s: watcher stop failed", self.spec.name)

    # ------------------------------------------------------- scheduler (D3)
    def start_schedule(self) -> bool:
        """Arm the auto-invoke timer if the spec carries an enabled schedule."""
        schedule = self.spec.schedule
        if schedule is None or not schedule.enabled or self._scheduler_task is not None:
            return False
        self._scheduler_task = asyncio.create_task(self._scheduler_loop(schedule.interval_s))
        logger.info(
            "agent %s: scheduled auto-invoke every %.0fs", self.spec.name, schedule.interval_s
        )
        return True

    def stop_schedule(self) -> None:
        task, self._scheduler_task = self._scheduler_task, None
        if task is not None:
            task.cancel()

    async def _scheduler_loop(self, interval_s: float) -> None:
        """Sleep, fire, repeat. One run's failure never kills the loop."""
        try:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    await self.fire_scheduled()
                except Exception:  # noqa: BLE001 — keep the schedule alive
                    logger.exception("agent %s: scheduled run failed", self.spec.name)
        except asyncio.CancelledError:
            pass

    async def fire_scheduled(self) -> RunRecord | None:
        """Run the scheduled input once. Skips (returns None) when not ready."""
        schedule = self.spec.schedule
        if schedule is None:
            return None
        ok, reason = self.ready
        if not ok:
            logger.warning("agent %s: skipping scheduled run — not ready (%s)", self.spec.name, reason)
            return None
        record = await self.start_run(schedule.input, wait=False)
        self._schedule_fire_count += 1
        self._schedule_last_fired = datetime.now(UTC)
        self._schedule_last_run_id = record.run_id
        return record

    def schedule_status(self) -> ScheduleStatus:
        schedule = self.spec.schedule
        return ScheduleStatus(
            configured=schedule is not None,
            enabled=bool(schedule and schedule.enabled),
            interval_s=schedule.interval_s if schedule else None,
            input=schedule.input if schedule else None,
            fire_count=self._schedule_fire_count,
            last_fired_at=self._schedule_last_fired,
            last_run_id=self._schedule_last_run_id,
        )

    def _on_source_event(self, _record: Any = None) -> None:
        """Watcher-thread callback: schedule a reload on the agent's event loop.

        The delivered record is ignored on purpose — reload() re-fetches via
        the source so cold load and hot-reload share one (preflighted,
        swap-safe) path.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.reload(), loop)

    def _parse_and_preflight(self, snapshot: ChainSnapshot) -> tuple[Any, list[str], list[str]]:
        from mmar_carl import ReasoningChain

        # C6/G9: fill in default chain-level + per-step timeouts (never looser
        # than the author's). Cold load and hot-reload share this path.
        content = inject_default_timeouts(
            snapshot.content, step_timeout_s=self.spec.step_timeout_s
        )
        chain = ReasoningChain.from_dict(content, use_typed_steps=True)
        probe = self._build_context("preflight probe", api=_NullLLM())
        report = chain.preflight(probe)
        required = list(report.required_tools or [])
        missing = [f"tool:{name}" for name in report.missing_tools or []]
        missing += [f"mcp:{name}" for name in report.missing_mcp_servers or []]
        missing += [f"skill:{name}" for name in report.missing_skills or []]
        return chain, required, missing

    # ----------------------------------------------------------------- state
    @property
    def ready(self) -> tuple[bool, str]:
        if self.chain is None:
            return False, self.load_error or "chain is not loaded"
        if self.missing:
            return False, "missing dependencies: " + ", ".join(self.missing)
        if self._llm is None:
            try:
                build_llm_client(self.spec)
            except LLMNotConfiguredError as exc:
                return False, str(exc)
        return True, "ok"

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = build_llm_client(self.spec)
        return self._llm

    def _build_context(self, task_input: str, *, api: Any | None = None) -> Any:
        from mmar_carl import Language, ReasoningContext

        language = Language.RUSSIAN if self.spec.language == "ru" else Language.ENGLISH
        ctx = ReasoningContext(outer_context=task_input, api=api or self._get_llm(), language=language)
        register_builtin_tools(ctx, web_search_api_key=os.environ.get(ENV_WEB_SEARCH_KEY))
        return ctx

    def _render_input(self, user_input: str) -> str:
        template = self.spec.task_template
        if template and "{input}" in template:
            return template.replace("{input}", user_input)
        return user_input

    def _get_recorder(self) -> RunRecorder | None:
        """Run-records go to Memory only in attached mode (there is an entity to
        link to and a client to write with). Built lazily from the source's client."""
        if not self._recorder_resolved:
            self._recorder_resolved = True
            if self.spec.entity_id and isinstance(self._source, MemoryChainSource):
                try:
                    self._recorder = RunRecorder(
                        self._source.get_client(),
                        entity_id=self.spec.entity_id,
                        agent_name=self.spec.name,
                    )
                except Exception as exc:
                    logger.warning("agent %s: run-recorder unavailable (%s)", self.spec.name, exc)
        return self._recorder

    # ----------------------------------------------------------------- runs
    async def start_run(self, user_input: str, *, wait: bool) -> RunRecord:
        """Start one chain run. ``wait=True`` (sync invoke) awaits completion;
        ``wait=False`` (async invoke) returns the running record immediately."""
        record = RunRecord(input=user_input)
        handle = RunHandle(record)
        self._remember(record, handle)
        async with self._swap_lock:
            chain = self.chain
        if chain is None:  # guarded by the endpoint; hard backstop
            record.status = "failed"
            record.success = False
            record.error = self.load_error or "chain is not loaded"
            record.finished_at = datetime.now(UTC)
            await handle.emit("result", record.model_dump(mode="json"))
            return record
        context = self._build_context(self._render_input(user_input))
        handle.context = context
        context.on_human_input_requested = self._make_human_input_handler(handle)
        handle.task = asyncio.create_task(self._execute(handle, chain, context))
        if wait:
            await asyncio.shield(handle.task)
        return record

    async def chat(self, message: str, session_id: str | None) -> tuple[str, RunRecord, int]:
        """One conversational turn (history-in-context, C3).

        The session's prior turns are rendered into the chain input; the chain
        runs synchronously; a successful exchange is appended to the session.
        Returns ``(session_id, run_record, turn_count)``. The chain is
        unchanged — only its ``outer_context`` carries the conversation.
        """
        session = self.sessions.get_or_create(session_id)
        composed = compose_chat_input(session.turns, message)
        record = await self.start_run(composed, wait=True)
        if record.success and record.answer is not None:
            updated = self.sessions.append_turn(session.session_id, message, record.answer)
            turn_count = len(updated.turns)
        else:
            turn_count = len(session.turns)
        return session.session_id, record, turn_count

    # ------------------------------------------------------ human input (D)
    def _make_human_input_handler(self, handle: RunHandle) -> Callable[[str, Any], None]:
        """Build the ``on_human_input_requested`` callback for one run.

        When the chain hits a ``human_input`` step CARL invokes this with the
        prompt + a future, then awaits the future. We pause the run: status →
        ``waiting``, expose the prompt on the record, snapshot the context (for
        durable cross-process resume), and emit an ``awaiting_input`` SSE event.
        ``POST /runs/{id}/input`` later resolves the future via
        ``provide_input``. Pause/resume is an async-invoke flow — a sync invoke
        would just block until the chain deadline.
        """

        def handler(prompt: str, future: Any) -> None:
            # The executor runs steps on a COPY of the context, so its
            # provide_human_input() future isn't reachable on handle.context.
            # The callback hands us the future directly — resolve THAT to resume.
            handle.input_future = future
            handle.awaiting_input = True
            handle.record.status = "waiting"
            handle.record.awaiting_input = {"prompt": prompt}
            try:
                if handle.context is not None:
                    handle.snapshot = handle.context.snapshot()
                    self._persist_snapshot(handle)
            except Exception:  # noqa: BLE001 — durability is best-effort
                logger.warning(
                    "agent %s: snapshot on pause failed", self.spec.name, exc_info=True
                )
            asyncio.ensure_future(
                handle.emit("awaiting_input", {"run_id": handle.record.run_id, "prompt": prompt})
            )

        return handler

    def _persist_snapshot(self, handle: RunHandle) -> None:
        """Best-effort: write the paused run's ContextSnapshot to disk when a
        ``snapshot_dir`` is configured (durable resume primitive)."""
        if not self.spec.snapshot_dir or handle.snapshot is None:
            return
        directory = Path(self.spec.snapshot_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{handle.record.run_id}.json"
        payload = json.dumps(handle.snapshot.model_dump(mode="json"), ensure_ascii=False)
        path.write_text(payload, encoding="utf-8")

    async def provide_input(self, run_id: str, value: str) -> RunRecord:
        """Resume a waiting run by answering its human-input step.

        Raises KeyError (→404) for an unknown run, ValueError (→409) when the
        run is not actually waiting / nothing is pending.
        """
        handle = self.handles.get(run_id)
        if handle is None:
            raise KeyError(run_id)
        future = handle.input_future
        if handle.record.status != "waiting" or future is None or future.done():
            raise ValueError(f"run is {handle.record.status}, not waiting for input")
        future.set_result(value)
        handle.input_future = None
        handle.awaiting_input = False
        handle.record.awaiting_input = None
        handle.record.status = "running"
        await handle.emit("input_provided", {"run_id": run_id})
        return handle.record

    async def _execute(self, handle: RunHandle, chain: Any, context: Any) -> None:
        """Drive one run via ``stream_async``: step events feed SSE, the terminal
        ReasoningResult fills the record. Hard deadline = spec.chain_timeout_s."""
        record = handle.record
        final: Any | None = None
        try:
            async with asyncio.timeout(self.spec.chain_timeout_s):
                async for item in chain.stream_async(context):
                    if _is_terminal_result(item):
                        final = item
                        continue
                    summary = _summarize_step(item)
                    record.steps.append(summary)
                    await handle.emit("step", summary.model_dump(mode="json"))
        except TimeoutError:
            handle.request_cancel()
            record.status = "timeout"
            record.success = False
            record.error = f"chain run exceeded {self.spec.chain_timeout_s:.0f}s"
        except Exception as exc:
            record.status = "failed"
            record.success = False
            record.error = f"{type(exc).__name__}: {exc}"
        else:
            if final is not None:
                record.success = bool(final.success)
                record.status = "succeeded" if record.success else "failed"
                record.answer = final.get_final_output()
                record.error = getattr(final, "error_message", None)
                record.token_usage = {k: int(v) for k, v in (getattr(final, "token_usage", {}) or {}).items()}
                record.execution_time_s = getattr(final, "execution_time", None)
            else:
                record.status = "failed"
                record.success = False
                record.error = record.error or "chain produced no result"
            if handle.cancel_requested:
                record.status = "cancelled"
                record.success = False
                record.error = record.error or "run cancelled"
        finally:
            if handle.cancel_requested and record.status not in ("succeeded",):
                record.status = "cancelled" if record.status != "timeout" else "timeout"
            handle.awaiting_input = False
            record.awaiting_input = None  # cleared at any terminal state
            record.finished_at = datetime.now(UTC)
            await handle.emit("result", record.model_dump(mode="json"))
            # after the result event: SSE consumers never wait on Memory I/O
            recorder = self._get_recorder()
            if recorder is not None:
                await recorder.record(record)

    def _remember(self, record: RunRecord, handle: RunHandle) -> None:
        self.runs[record.run_id] = record
        self.handles[record.run_id] = handle
        while len(self.runs) > _MAX_RUNS_KEPT:
            old_id, _ = self.runs.popitem(last=False)
            self.handles.pop(old_id, None)


def _is_terminal_result(item: Any) -> bool:
    """stream_async yields StepExecutionResult per step, then one ReasoningResult."""
    return hasattr(item, "step_results")


def _summarize_step(step: Any) -> StepSummary:
    return StepSummary(
        number=getattr(step, "step_number", None) or getattr(step, "number", None),
        title=str(getattr(step, "step_title", "") or getattr(step, "title", "") or ""),
        step_type=str(getattr(step, "step_type", "") or ""),
        success=getattr(step, "success", None),
        error=getattr(step, "error_message", None),
    )
