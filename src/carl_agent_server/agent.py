"""AgentState — one chain behind an HTTP facade: load, preflight, runs (sync/async/SSE/cancel)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

from .chain_source import ChainSnapshot, ChainSource, FileChainSource, MemoryChainSource
from .llm import LLMNotConfiguredError, build_llm_client
from .models import AgentMeta, DeploymentSpec, RunRecord, StepSummary
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
    ) -> None:
        self.spec = spec
        self._source: ChainSource = chain_source or self._default_source(spec)
        self._llm = llm_client
        self.chain: Any | None = None
        self.meta: AgentMeta = AgentMeta(display_name=spec.name)
        self.required_tools: list[str] = []
        self.missing: list[str] = []
        self.load_error: str | None = None
        self.loaded_at: datetime | None = None
        self.runs: OrderedDict[str, RunRecord] = OrderedDict()
        self.handles: dict[str, RunHandle] = {}
        self.on_reloaded: list[Callable[[], None]] = []
        self._swap_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
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

        chain = ReasoningChain.from_dict(dict(snapshot.content), use_typed_steps=True)
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
        handle.task = asyncio.create_task(self._execute(handle, chain, context))
        if wait:
            await asyncio.shield(handle.task)
        return record

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
            record.finished_at = datetime.now(UTC)
            await handle.emit("result", record.model_dump(mode="json"))

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
