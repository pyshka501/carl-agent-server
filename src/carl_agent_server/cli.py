"""carl-agent — solo-serve one chain as an HTTP agent.

Examples
--------
Serve a chain JSON from disk (offline):

    carl-agent serve --chain-file ./chain.json --name demo --port 8001

Serve a chain entity from gigaevo Memory, following its stable channel:

    AGENT_LLM_API_KEY=... AGENT_LLM_MODEL=openai/gpt-4o \\
    carl-agent serve --entity-id <uuid> --channel stable --port 8001
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from .app import build_agent_app
from .models import DeploymentSpec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="carl-agent", description="Serve ONE CARL chain as an HTTP agent")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the agent server")
    serve.add_argument("--name", default="agent", help="url-safe agent name")
    source = serve.add_mutually_exclusive_group(required=True)
    source.add_argument("--chain-file", help="path to a chain JSON (offline mode)")
    source.add_argument("--entity-id", help="Memory chain entity id (attached mode)")
    serve.add_argument("--channel", default="stable", help="Memory channel to follow (attached mode)")
    serve.add_argument("--memory-url", default=os.environ.get("AGENT_MEMORY_URL"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)

    args = parser.parse_args(argv)
    spec = DeploymentSpec(
        name=args.name,
        chain_file=args.chain_file,
        entity_id=args.entity_id,
        channel=args.channel,
        memory_url=args.memory_url,
    )
    uvicorn.run(build_agent_app(spec), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
