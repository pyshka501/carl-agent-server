"""CLI entrypoints: ``carl-agent`` (solo, one chain) and ``carl-agent-hub`` (N agents).

Examples
--------
Serve a chain JSON from disk (offline solo):

    carl-agent serve --chain-file ./chain.json --name demo --port 8001

Serve a chain entity from gigaevo Memory, following its stable channel:

    AGENT_LLM_API_KEY=... AGENT_LLM_MODEL=openai/gpt-4o \\
    carl-agent serve --entity-id <uuid> --channel stable --port 8001

Run the hub (deploy agents at runtime via its control API):

    carl-agent-hub serve --port 8080
    curl -X POST localhost:8080/deployments \\
         -H 'content-type: application/json' \\
         -d '{"name": "demo", "chain_file": "./chain.json"}'
    # -> open http://localhost:8080/agents/demo/docs
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from .app import build_agent_app
from .hub import DEFAULT_STATE_FILE, build_hub_app
from .models import DeploymentSpec


def main(argv: list[str] | None = None) -> None:
    """``carl-agent`` — solo-serve ONE chain as an HTTP agent."""
    parser = argparse.ArgumentParser(prog="carl-agent", description="Serve ONE CARL chain as an HTTP agent")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the agent server")
    serve.add_argument("--name", default="agent", help="url-safe agent name")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--chain-file", help="path to a chain JSON (offline mode)")
    source.add_argument("--entity-id", help="Memory chain entity id (attached mode)")
    serve.add_argument("--channel", default="stable", help="Memory channel to follow (attached mode)")
    serve.add_argument("--memory-url", default=os.environ.get("AGENT_MEMORY_URL"))
    serve.add_argument(
        "--api-key",
        default=os.environ.get("AGENT_API_KEY"),
        help="Require this API key on /invoke /chat /runs (env: AGENT_API_KEY).",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)

    args = parser.parse_args(argv)
    spec = DeploymentSpec(
        name=args.name,
        chain_file=args.chain_file,
        entity_id=args.entity_id,
        channel=args.channel,
        memory_url=args.memory_url,
        api_key=args.api_key,
    )
    uvicorn.run(build_agent_app(spec), host=args.host, port=args.port)


def hub_main(argv: list[str] | None = None) -> None:
    """``carl-agent-hub`` — serve N agents in one process (control API + /agents/<name>)."""
    parser = argparse.ArgumentParser(
        prog="carl-agent-hub",
        description="Agent hub: control API at /, agents (each with its own /docs) at /agents/<name>",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the hub server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        help=f"where deployments persist between restarts (default: {DEFAULT_STATE_FILE})",
    )
    serve.add_argument(
        "--no-persist",
        action="store_true",
        help="do not persist deployments (a restart starts empty)",
    )

    args = parser.parse_args(argv)
    state_file = None if args.no_persist else args.state_file
    uvicorn.run(build_hub_app(state_file=state_file), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
