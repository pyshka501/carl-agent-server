"""carl-agent-server — serve CARL reasoning chains as HTTP agents.

One chain = one agent: a FastAPI facade with /invoke, /info, /healthz and its
OWN /docs (OpenAPI metadata is taken from the chain's Memory entity, so the
Swagger page reads as that agent's documentation). Agents are served solo
(`carl-agent serve`) or mounted together in the hub (`/agents/<name>/…`).
"""

from .app import build_agent_app
from .hub import build_hub_app
from .models import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    DeploymentInfo,
    DeploymentSpec,
    HumanInputRequest,
    InvokeRequest,
    RunRecord,
)

__version__ = "0.1.0"

__all__ = [
    "AgentInfo",
    "ChatRequest",
    "ChatResponse",
    "DeploymentInfo",
    "DeploymentSpec",
    "HumanInputRequest",
    "InvokeRequest",
    "RunRecord",
    "__version__",
    "build_agent_app",
    "build_hub_app",
]
