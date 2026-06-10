"""AgentState — one chain behind an HTTP facade: load, preflight, invoke, run store."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
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
        self._swap_lock = asyncio.Lock()
        self._loaded_once = False

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

        On any failure the previous chain (if any) stays serving.
        """
        self._loaded_once = True
        try:
            snapshot = await asyncio.to_thread(self._source.load)
            chain, required, missing = self._parse_and_preflight(snapshot)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("agent %s: reload failed, keeping previous chain (%s)", self.spec.name, self.load_error)
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

    def _parse_and_preflight(self, snapshot: ChainSnapshot) -> tuple[Any, list[str], list[str]]:
        from mmar_carl import ReasoningChain

        chain = ReasoningChain.from_dict(dict(snapshot.content), use_typed_steps=True)
        probe = self._build_context("preflight probe", api=_NullLLM())
        report = chain.preflight(probe)
        required = list(report.required_tools or [])
        missing = [f"tool:{t}" for t in report.missing_tools or []]
        missing += [f"mcp:{s}" for s in report.missing_mcp_servers or []]
        missing += [f"skill:{s}" for s in report.missing_skills or []]
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
    async def invoke(self, user_input: str) -> RunRecord:
        """Execute one chain run with the deployment's hard timeout; always returns a record."""
        record = RunRecord(input=user_input)
        self._remember(record)
        async with self._swap_lock:
            chain = self.chain
        if chain is None:  # guarded by the endpoint, kept as a hard backstop
            record.status = "failed"
            record.success = False
            record.error = self.load_error or "chain is not loaded"
            record.finished_at = datetime.now(UTC)
            return record
        ctx = self._build_context(self._render_input(user_input))
        try:
            result = await asyncio.wait_for(chain.execute_async(ctx), timeout=self.spec.chain_timeout_s)
        except TimeoutError:
            record.status = "timeout"
            record.success = False
            record.error = f"chain run exceeded {self.spec.chain_timeout_s:.0f}s"
        except Exception as exc:
            record.status = "failed"
            record.success = False
            record.error = f"{type(exc).__name__}: {exc}"
        else:
            record.success = bool(result.success)
            record.status = "succeeded" if record.success else "failed"
            record.answer = result.get_final_output()
            record.error = getattr(result, "error_message", None)
            record.token_usage = {k: int(v) for k, v in (getattr(result, "token_usage", {}) or {}).items()}
            record.execution_time_s = getattr(result, "execution_time", None)
            record.steps = [_summarize_step(s) for s in getattr(result, "step_results", []) or []]
        record.finished_at = datetime.now(UTC)
        return record

    def _remember(self, record: RunRecord) -> None:
        self.runs[record.run_id] = record
        while len(self.runs) > _MAX_RUNS_KEPT:
            self.runs.popitem(last=False)


def _summarize_step(step: Any) -> StepSummary:
    return StepSummary(
        number=getattr(step, "step_number", None) or getattr(step, "number", None),
        title=str(getattr(step, "step_title", "") or getattr(step, "title", "") or ""),
        step_type=str(getattr(step, "step_type", "") or ""),
        success=getattr(step, "success", None),
        error=getattr(step, "error_message", None),
    )
