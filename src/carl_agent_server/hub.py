"""The agent hub — N agents in one process, each with its OWN Swagger.

Deploy = build the agent app from a :class:`DeploymentSpec`, activate it
(load + preflight + hot-reload watcher) and mount it under ``/agents/<name>``.
FastAPI serves a mounted sub-application's docs at ``/agents/<name>/docs``,
so every agent keeps a personal Swagger page; the hub's own ``/docs`` covers
the control API only.

Deployment specs persist to a JSON state file and are re-deployed on startup,
so a hub restart restores its agents. NOTE: the spec is persisted verbatim —
prefer env vars (``AGENT_LLM_API_KEY`` …) over inline ``llm_api_key`` so
secrets stay out of the state file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from starlette.routing import Mount

from .agent import AgentState
from .app import activate_agent_app, build_agent_app, deactivate_agent_app
from .models import DeploymentInfo, DeploymentSpec

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "~/.care/agent-hub.json"

AgentAppFactory = Callable[[DeploymentSpec], FastAPI]


class _Deployment:
    def __init__(self, spec: DeploymentSpec, app: FastAPI) -> None:
        self.spec = spec
        self.app = app
        self.deployed_at = datetime.now(UTC)

    @property
    def state(self) -> AgentState:
        return self.app.state.agent


class AgentHub:
    """Mount/unmount agents on a running root app + persist the deployment set."""

    def __init__(self, app: FastAPI, *, state_file: Path | None, factory: AgentAppFactory) -> None:
        self.app = app
        self.state_file = state_file
        self.factory = factory
        self.deployments: dict[str, _Deployment] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ operations
    async def deploy(self, spec: DeploymentSpec, *, persist: bool = True) -> DeploymentInfo:
        async with self._lock:
            if spec.name in self.deployments:
                raise HTTPException(status_code=409, detail=f"deployment {spec.name!r} already exists")
            agent_app = self.factory(spec)
            state = await activate_agent_app(agent_app)
            if state.chain is None:
                deactivate_agent_app(agent_app)
                raise HTTPException(status_code=422, detail=f"deploy failed: {state.load_error or 'chain not loaded'}")
            self.app.mount(f"/agents/{spec.name}", agent_app)
            deployment = _Deployment(spec, agent_app)
            self.deployments[spec.name] = deployment
            if persist:
                self._save_state()
            logger.info("hub: deployed %r -> /agents/%s (%s)", spec.name, spec.name, state.meta.version_label)
            return self._info(deployment)

    async def undeploy(self, name: str) -> None:
        async with self._lock:
            deployment = self.deployments.pop(name, None)
            if deployment is None:
                raise HTTPException(status_code=404, detail=f"deployment {name!r} not found")
            deactivate_agent_app(deployment.app)
            prefix = f"/agents/{name}"
            self.app.router.routes = [
                route
                for route in self.app.router.routes
                if not (isinstance(route, Mount) and route.path == prefix)
            ]
            self._save_state()
            logger.info("hub: undeployed %r", name)

    async def reload(self, name: str) -> tuple[bool, DeploymentInfo]:
        deployment = self.deployments.get(name)
        if deployment is None:
            raise HTTPException(status_code=404, detail=f"deployment {name!r} not found")
        reloaded = await deployment.state.reload()
        return reloaded, self._info(deployment)

    def list_info(self) -> list[DeploymentInfo]:
        return [self._info(d) for d in self.deployments.values()]

    def get_info(self, name: str) -> DeploymentInfo:
        deployment = self.deployments.get(name)
        if deployment is None:
            raise HTTPException(status_code=404, detail=f"deployment {name!r} not found")
        return self._info(deployment)

    # ------------------------------------------------------------- lifecycle
    async def restore(self) -> None:
        """Re-deploy everything from the state file (startup). Failures skip."""
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("hub: unreadable state file %s (%s); starting empty", self.state_file, exc)
            return
        for raw in data.get("deployments", []):
            try:
                spec = DeploymentSpec.model_validate(raw)
                await self.deploy(spec, persist=False)
            except Exception as exc:
                detail = getattr(exc, "detail", exc)
                logger.warning("hub: could not restore deployment %r: %s", raw.get("name", "?"), detail)

    async def shutdown(self) -> None:
        for deployment in self.deployments.values():
            deactivate_agent_app(deployment.app)

    # --------------------------------------------------------------- helpers
    def _info(self, deployment: _Deployment) -> DeploymentInfo:
        state = deployment.state
        ok, reason = state.ready
        meta = state.meta
        spec = deployment.spec
        return DeploymentInfo(
            name=spec.name,
            url=f"/agents/{spec.name}",
            display_name=meta.display_name,
            version=meta.version_label,
            ready=ok,
            ready_reason=reason,
            entity_id=spec.entity_id,
            channel=spec.channel if spec.entity_id else None,
            chain_file=spec.chain_file,
            source=meta.source,
            deployed_at=deployment.deployed_at,
            runs=len(state.runs),
        )

    def _save_state(self) -> None:
        if self.state_file is None:
            return
        specs = [d.spec.model_dump(exclude_none=True) for d in self.deployments.values()]
        payload = json.dumps({"deployments": specs}, ensure_ascii=False, indent=2)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        # The state file holds per-agent API keys — keep it owner-only.
        try:
            os.chmod(tmp, 0o600)
        except OSError:  # pragma: no cover - non-POSIX
            logger.debug("hub: could not chmod state file", exc_info=True)
        tmp.replace(self.state_file)


def build_hub_app(
    *,
    state_file: str | Path | None = DEFAULT_STATE_FILE,
    agent_app_factory: AgentAppFactory | None = None,
) -> FastAPI:
    """Build the hub: control API at the root, agents mounted at /agents/<name>.

    ``agent_app_factory`` is the injection point for tests / custom wiring;
    it defaults to :func:`build_agent_app`. ``state_file=None`` disables
    persistence.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await hub.restore()
        try:
            yield
        finally:
            await hub.shutdown()

    app = FastAPI(
        title="CARL Agent Hub",
        description=(
            "Control API for deployed chain-agents. Each deployment is mounted at "
            "`/agents/<name>` and serves its **own** Swagger at `/agents/<name>/docs`."
        ),
        lifespan=lifespan,
    )
    resolved_state = Path(state_file).expanduser() if state_file is not None else None
    hub = AgentHub(app, state_file=resolved_state, factory=agent_app_factory or build_agent_app)
    app.state.hub = hub

    @app.get("/healthz", tags=["hub"])
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "deployments": len(hub.deployments)}

    @app.get("/deployments", response_model=list[DeploymentInfo], tags=["hub"])
    async def list_deployments() -> list[DeploymentInfo]:
        return hub.list_info()

    @app.get("/deployments/{name}", response_model=DeploymentInfo, tags=["hub"])
    async def get_deployment(name: str) -> DeploymentInfo:
        return hub.get_info(name)

    @app.post("/deployments", response_model=DeploymentInfo, status_code=201, tags=["hub"])
    async def create_deployment(spec: DeploymentSpec) -> DeploymentInfo:
        return await hub.deploy(spec)

    @app.delete("/deployments/{name}", tags=["hub"])
    async def delete_deployment(name: str) -> dict[str, str]:
        await hub.undeploy(name)
        return {"deleted": name}

    @app.post("/deployments/{name}/reload", tags=["hub"])
    async def reload_deployment(name: str) -> dict[str, Any]:
        reloaded, info = await hub.reload(name)
        return {"reloaded": reloaded, "deployment": info.model_dump(mode="json")}

    return app
