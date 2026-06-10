"""build_agent_app — the FastAPI facade for ONE chain, with its own /docs.

The app's OpenAPI metadata (title / description / version) is refreshed from
the chain's metadata after load, so /docs reads as THIS agent's documentation:
the entity's display name becomes the title, its description + when-to-use +
example task become the description, the resolved Memory version becomes the
API version. Mounted as a sub-application (the hub does this) FastAPI serves
the same /docs under /agents/<name>/docs out of the box.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import AgentState
from .models import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    DeploymentSpec,
    InvokeRequest,
    RunRecord,
)


def build_agent_app(
    spec: DeploymentSpec,
    *,
    chain_source: Any | None = None,
    llm_client: Any | None = None,
    run_recorder: Any | None = None,
) -> FastAPI:
    """Build the FastAPI app serving one deployed chain.

    ``chain_source`` / ``llm_client`` / ``run_recorder`` are injection points
    for the hub, solo CLI and tests; by default the source comes from the spec
    (file or Memory), the LLM client from spec/env at first use, and the
    run-recorder from the Memory source's client (attached mode only).
    """
    state = AgentState(spec, chain_source=chain_source, llm_client=llm_client, run_recorder=run_recorder)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await activate_agent_app(app)
        try:
            yield
        finally:
            deactivate_agent_app(app)

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
    async def invoke(request: InvokeRequest, mode: Literal["sync", "async"] = "sync") -> Any:
        """Run the chain. ``mode=sync`` waits for the result; ``mode=async``
        returns 202 with a running record — poll /runs/{id} or stream
        /runs/{id}/events."""
        await state.ensure_loaded()
        ok, reason = state.ready
        if not ok:
            raise HTTPException(status_code=503, detail=f"agent is not ready: {reason}")
        if mode == "async":
            record = await state.start_run(request.input, wait=False)
            return JSONResponse(record.model_dump(mode="json"), status_code=202)
        return await state.start_run(request.input, wait=True)

    @app.post("/chat", response_model=ChatResponse, tags=["agent"])
    async def chat(request: ChatRequest) -> ChatResponse:
        """Hold a conversation with the agent. Omit ``session_id`` to start a
        new session (the reply carries the id); pass it back to continue — the
        dialogue so far is fed into the chain each turn (the chain is
        unchanged). Sessions evict after the deployment's idle TTL."""
        await state.ensure_loaded()
        ok, reason = state.ready
        if not ok:
            raise HTTPException(status_code=503, detail=f"agent is not ready: {reason}")
        session_id, record, turn_count = await state.chat(request.message, request.session_id)
        return ChatResponse(
            session_id=session_id,
            run_id=record.run_id,
            status=record.status,
            answer=record.answer,
            error=record.error,
            turn_count=turn_count,
        )

    @app.get("/runs/{run_id}", response_model=RunRecord, tags=["agent"])
    async def get_run(run_id: str) -> RunRecord:
        record = state.runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return record

    @app.get("/runs/{run_id}/events", tags=["agent"])
    async def run_events(run_id: str) -> StreamingResponse:
        """Server-Sent Events: replays completed steps, tails live ones, ends
        with the terminal ``result`` event. Disconnecting stops the stream but
        does NOT cancel the run — use DELETE /runs/{id} for that."""
        handle = state.handles.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")

        async def gen() -> Any:
            async for event in handle.stream():
                payload = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event['event']}\ndata: {payload}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.delete("/runs/{run_id}", tags=["agent"])
    async def cancel_run(run_id: str) -> dict[str, str]:
        """Cooperatively cancel a running run (CARL stops between step batches)."""
        handle = state.handles.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        if handle.record.status != "running":
            raise HTTPException(status_code=409, detail=f"run is already {handle.record.status}")
        handle.request_cancel()
        return {"run_id": run_id, "status": "cancelling"}

    return app


async def activate_agent_app(app: FastAPI) -> AgentState:
    """Run the agent app's startup work: load + preflight, /docs metadata,
    hot-reload watcher.

    The solo path runs this from the app's own lifespan; the HUB calls it
    explicitly for mounted sub-applications, whose lifespans Starlette does
    not run. Idempotent — re-activation only re-ensures the load.
    """
    state: AgentState = app.state.agent
    if getattr(app.state, "activated", False):
        await state.ensure_loaded()
        return state
    app.state.activated = True
    # the hook keeps /docs metadata in sync across hot-reloads (A3)
    state.on_reloaded.append(lambda: refresh_app_meta(app, state))
    await state.ensure_loaded()
    refresh_app_meta(app, state)
    state.start_watch()
    return state


def deactivate_agent_app(app: FastAPI) -> None:
    """Stop the agent's background work (the watcher). Safe to call twice."""
    app.state.activated = False
    app.state.agent.stop_watch()


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
