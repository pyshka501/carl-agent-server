"""Pydantic models: deployment spec, agent metadata, run records, API shapes."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


class ScheduleConfig(BaseModel):
    """Auto-invoke the agent's chain on a fixed cadence (D3).

    A simple interval timer — external cron/batch is served by ``care run``;
    this is the in-template scheduler so a deployment can self-trigger.
    """

    interval_s: float = Field(..., gt=0, description="Seconds between automatic runs")
    input: str = Field(..., min_length=1, description="The task fed to the chain each tick")
    enabled: bool = Field(default=True, description="Set false to keep the schedule but pause firing")


class BundledTool(BaseModel):
    """A synthesized tool shipped WITH a deployment so the agent can call it.

    The deploying client (CARE) generates these tools locally; the hub doesn't
    ship them in its read-only builtin set. ``source`` is a standalone Python
    module defining ``def <name>(**kwargs)``. The hub runs it in an isolated
    subprocess per call (no in-process exec), so a tool can't corrupt the hub.
    """

    name: str = Field(..., description="Tool name the chain's tool steps call")
    source: str = Field(..., description="Python source defining `def <name>(**kwargs) -> str`")
    params: list[str] = Field(default_factory=list, description="Declared parameter names (advisory)")
    description: str = Field(default="", description="What the tool does (advisory)")


class DeploymentSpec(BaseModel):
    """One agent = one chain + how to run it.

    Exactly one chain source is required: ``entity_id`` (attached mode — the
    chain is fetched from gigaevo Memory and follows ``channel``) or
    ``chain_file`` (offline mode — a chain JSON on disk; solo serving / tests).
    """

    name: str = Field(..., description="URL-safe agent name; the hub mounts it at /agents/<name>")
    extra_tools: list[BundledTool] = Field(
        default_factory=list,
        description="Synthesized tools shipped with the chain (not in the builtin set), "
        "registered into the agent's context and run in an isolated subprocess.",
    )
    entity_id: str | None = Field(default=None, description="Memory chain entity id (attached mode)")
    channel: str = Field(default="stable", description="Memory channel the agent follows (attached mode)")
    chain_file: str | None = Field(default=None, description="Path to a chain JSON file (offline mode)")
    task_template: str | None = Field(
        default=None,
        description="Optional input template; '{input}' is replaced with the request input",
    )
    language: Literal["en", "ru"] = "en"

    llm_model: str | None = Field(default=None, description="Model id; falls back to AGENT_LLM_MODEL")
    llm_base_url: str | None = Field(default=None, description="OpenAI-compatible base URL; falls back to AGENT_LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, description="API key; falls back to AGENT_LLM_API_KEY")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    chain_timeout_s: float = Field(default=300.0, gt=0, description="Hard deadline for one chain run")
    step_timeout_s: float = Field(
        default=60.0,
        gt=0,
        description=(
            "Default per-step timeout injected at load into steps without one "
            "(capped never to exceed the chain-level timeout). Bounds a single "
            "hung step well inside chain_timeout_s."
        ),
    )

    session_ttl_s: float = Field(
        default=1800.0, gt=0, description="Idle lifetime of a /chat session before eviction"
    )
    session_history_turns: int = Field(
        default=6, ge=0, description="Prior turns fed into the chain on each /chat message"
    )

    api_key: str | None = Field(
        default=None,
        description=(
            "Per-agent API key. When set, /invoke /chat and /runs require it via "
            "`X-API-Key` (or `Authorization: Bearer`); /healthz /readyz /info /docs "
            "stay open. None disables auth (localhost demo)."
        ),
    )
    auth_allow_localhost: bool = Field(
        default=True,
        description="Loopback (127.0.0.1/::1) requests skip the API-key check — for local dev/demo.",
    )
    snapshot_dir: str | None = Field(
        default=None,
        description="Directory to persist a paused run's ContextSnapshot (durable "
        "human-input pause/resume). None = in-memory only.",
    )
    schedule: ScheduleConfig | None = Field(
        default=None,
        description="Auto-invoke the chain on a fixed interval (D3). None = no schedule.",
    )

    price_per_1k_input_usd: float | None = Field(
        default=None, ge=0, description="USD per 1k input tokens (cost tracking; D4)"
    )
    price_per_1k_output_usd: float | None = Field(
        default=None, ge=0, description="USD per 1k output tokens (cost tracking; D4)"
    )
    budget_usd: float | None = Field(
        default=None,
        gt=0,
        description="Total USD spend cap; further runs are refused (402) once reached. "
        "Needs pricing to take effect.",
    )

    memory_url: str | None = Field(default=None, description="Memory API base URL; falls back to AGENT_MEMORY_URL")
    memory_api_key: str | None = Field(default=None, description="Memory API key; falls back to AGENT_MEMORY_API_KEY")
    poll_fallback_s: float = Field(
        default=60.0,
        ge=0,
        description=(
            "Attached mode: also poll the followed channel every N seconds and reload "
            "on a version change — safety net for when the SSE watcher is silently "
            "broken (unreachable/misrouted events endpoint). 0 disables polling."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> DeploymentSpec:
        if not _NAME_RE.match(self.name):
            raise ValueError("name must be url-safe: lowercase letters, digits, '.', '_' or '-' (max 63 chars)")
        if bool(self.entity_id) == bool(self.chain_file):
            raise ValueError("exactly one of entity_id (attached) or chain_file (offline) is required")
        return self


class AgentMeta(BaseModel):
    """Presentation metadata extracted from the chain source — drives the agent's OpenAPI/docs."""

    display_name: str
    description: str = ""
    when_to_use: str = ""
    example_task: str = ""
    version_label: str = "unloaded"
    version_id: str = ""
    entity_id: str | None = None
    channel: str | None = None
    source: Literal["memory", "file"] = "file"


class StepSummary(BaseModel):
    number: int | None = None
    title: str = ""
    step_type: str = ""
    success: bool | None = None
    error: str | None = None


RunStatus = Literal["running", "waiting", "succeeded", "failed", "timeout", "cancelled"]


class RunRecord(BaseModel):
    """One invocation of the agent's chain."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: RunStatus = "running"
    input: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    success: bool | None = None
    answer: str | None = None
    error: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = Field(default=None, description="Run cost in USD when pricing is configured (D4)")
    execution_time_s: float | None = None
    steps: list[StepSummary] = Field(default_factory=list)
    awaiting_input: dict[str, Any] | None = Field(
        default=None,
        description="When status is 'waiting': the pending human-input prompt. "
        "POST /runs/{id}/input to resume.",
    )


class InvokeRequest(BaseModel):
    input: str = Field(..., min_length=1, description="The task / question for the agent")


class HumanInputRequest(BaseModel):
    value: str = Field(..., description="The human's answer to resume a waiting run")


class ScheduleStatus(BaseModel):
    """The deployment's auto-invoke schedule and its firing stats (D3)."""

    configured: bool = False
    enabled: bool = False
    interval_s: float | None = None
    input: str | None = None
    fire_count: int = 0
    last_fired_at: datetime | None = None
    last_run_id: str | None = None


class MetricsReport(BaseModel):
    """Usage + cost metrics for one deployment (D4)."""

    run_count: int = 0
    status_counts: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float | None = None
    pricing_configured: bool = False
    budget_usd: float | None = None
    remaining_usd: float | None = None
    over_budget: bool = False


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's message in the conversation")
    session_id: str | None = Field(
        default=None,
        description="Continue an existing session; omit to start a new one (id returned in the reply)",
    )


class ChatResponse(BaseModel):
    """One conversational exchange with a deployed agent."""

    session_id: str
    run_id: str
    status: RunStatus
    answer: str | None = None
    error: str | None = None
    turn_count: int = 0


class DeploymentInfo(BaseModel):
    """One hub deployment as reported by the control API."""

    name: str
    url: str = Field(..., description="Mount path of the agent (its /docs lives under it)")
    display_name: str
    version: str
    ready: bool
    ready_reason: str
    entity_id: str | None = None
    channel: str | None = None
    chain_file: str | None = None
    source: str
    deployed_at: datetime
    runs: int = 0


class AgentInfo(BaseModel):
    """The agent card served at /info."""

    name: str
    display_name: str
    description: str = ""
    when_to_use: str = ""
    example_task: str = ""
    version: str
    channel: str | None = None
    entity_id: str | None = None
    source: str
    required_tools: list[str] = Field(default_factory=list)
    ready: bool
    ready_reason: str = "ok"
