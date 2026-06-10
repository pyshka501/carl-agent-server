"""Pydantic models: deployment spec, agent metadata, run records, API shapes."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


class DeploymentSpec(BaseModel):
    """One agent = one chain + how to run it.

    Exactly one chain source is required: ``entity_id`` (attached mode — the
    chain is fetched from gigaevo Memory and follows ``channel``) or
    ``chain_file`` (offline mode — a chain JSON on disk; solo serving / tests).
    """

    name: str = Field(..., description="URL-safe agent name; the hub mounts it at /agents/<name>")
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

    memory_url: str | None = Field(default=None, description="Memory API base URL; falls back to AGENT_MEMORY_URL")
    memory_api_key: str | None = Field(default=None, description="Memory API key; falls back to AGENT_MEMORY_API_KEY")

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
    entity_id: str | None = None
    channel: str | None = None
    source: Literal["memory", "file"] = "file"


class StepSummary(BaseModel):
    number: int | None = None
    title: str = ""
    step_type: str = ""
    success: bool | None = None
    error: str | None = None


RunStatus = Literal["running", "succeeded", "failed", "timeout", "cancelled"]


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
    execution_time_s: float | None = None
    steps: list[StepSummary] = Field(default_factory=list)


class InvokeRequest(BaseModel):
    input: str = Field(..., min_length=1, description="The task / question for the agent")


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
