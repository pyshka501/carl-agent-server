"""build_agent_app — the FastAPI facade for ONE chain, with its own /docs.

The app's OpenAPI metadata (title / description / version) is refreshed from
the chain's metadata after load, so /docs reads as THIS agent's documentation:
the entity's display name becomes the title, its description + when-to-use +
example task become the description, the resolved Memory version becomes the
API version. Mounted as a sub-application (the hub does this) FastAPI serves
the same /docs under /agents/<name>/docs out of the box.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .agent import AgentState
from .models import AgentInfo, DeploymentSpec, InvokeRequest, RunRecord


def build_agent_app(
    spec: DeploymentSpec,
    *,
    chain_source: Any | None = None,
    llm_client: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app serving one deployed chain.

    ``chain_source`` / ``llm_client`` are injection points for the hub, solo
    CLI and tests; by default the source comes from the spec (file or Memory)
    and the LLM client from spec/env at first use.
    """
    state = AgentState(spec, chain_source=chain_source, llm_client=llm_client)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.ensure_loaded()
        refresh_app_meta(app, state)
        yield

    app = FastAPI(
        title=spec.name,
        version="unloaded",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.agent = state

    @app.get("/healthz", tags=["service"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "agent": spec.name}

    @app.get("/readyz", tags=["service"])
    async def readyz() -> JSONResponse:
        await state.ensure_loaded()
        ok, reason = state.ready
        body = {"ready": ok, "reason": reason, "version": state.meta.version_label}
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/info", response_model=AgentInfo, tags=["service"])
    async def info() -> AgentInfo:
        await state.ensure_loaded()
        ok, reason = state.ready
        meta = state.meta
        return AgentInfo(
            name=spec.name,
            display_name=meta.display_name,
            description=meta.description,
            when_to_use=meta.when_to_use,
            example_task=meta.example_task,
            version=meta.version_label,
            channel=meta.channel,
            entity_id=meta.entity_id,
            source=meta.source,
            required_tools=state.required_tools,
            ready=ok,
            ready_reason=reason,
        )

    @app.post("/invoke", response_model=RunRecord, tags=["agent"])
    async def invoke(request: InvokeRequest) -> RunRecord:
        await state.ensure_loaded()
        ok, reason = state.ready
        if not ok:
            raise HTTPException(status_code=503, detail=f"agent is not ready: {reason}")
        return await state.invoke(request.input)

    @app.get("/runs/{run_id}", response_model=RunRecord, tags=["agent"])
    async def get_run(run_id: str) -> RunRecord:
        record = state.runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return record

    return app


def refresh_app_meta(app: FastAPI, state: AgentState) -> None:
    """Re-point the app's OpenAPI metadata at the loaded chain's metadata."""
    meta = state.meta
    app.title = meta.display_name or app.title
    parts = [meta.description.strip()] if meta.description.strip() else []
    if meta.when_to_use.strip():
        parts.append(f"**When to use:** {meta.when_to_use.strip()}")
    if meta.example_task.strip():
        parts.append(f"**Example task:** {meta.example_task.strip()}")
    parts.append(f"_Source: {meta.source}" + (f" · channel `{meta.channel}`_" if meta.channel else "_"))
    app.description = "\n\n".join(parts)
    app.version = meta.version_label
    app.openapi_schema = None  # drop the cached schema so /openapi.json rebuilds
