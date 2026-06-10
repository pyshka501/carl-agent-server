# carl-agent-server

Serve [CARL](https://github.com/Glazkoff/carl) reasoning chains as HTTP
**agents**. One chain = one agent: a FastAPI facade with `/invoke`, `/info`,
`/healthz` and its **own Swagger** (`/docs`) — the OpenAPI title, description
and version come from the chain's metadata, so the docs page reads as *that
agent's* documentation and users can try the agent right from `/docs`.

Part of the MMAR ecosystem: chains are authored in
[CARE](https://github.com/Glazkoff/care) / [MAGE](https://github.com/Glazkoff/carl-mage),
stored & versioned in gigaevo Memory (channels: `latest` → `stable`), executed
by [mmar-carl](https://pypi.org/project/mmar-carl/). In *attached* mode the
agent follows a Memory channel and hot-reloads on `promote`/`pin` — promote a
new version and the running agent picks it up with zero downtime; pin the old
version to roll back.

## Quickstart (offline: a chain JSON from disk)

```bash
uv sync --group dev

AGENT_LLM_API_KEY=sk-... AGENT_LLM_MODEL=openai/gpt-4o \
uv run carl-agent serve --chain-file ./chain.json --name demo --port 8001
# open http://127.0.0.1:8001/docs and try POST /invoke
```

## Attached mode (gigaevo Memory)

```bash
AGENT_LLM_API_KEY=... AGENT_LLM_MODEL=... AGENT_MEMORY_URL=http://localhost:8002 \
uv run carl-agent serve --entity-id <chain-entity-uuid> --channel stable --port 8001
```

## Environment

| Variable | Purpose |
|---|---|
| `AGENT_LLM_API_KEY` / `AGENT_LLM_MODEL` / `AGENT_LLM_BASE_URL` | OpenAI-compatible LLM the chain runs on |
| `AGENT_MEMORY_URL` / `AGENT_MEMORY_API_KEY` | gigaevo Memory (attached mode) |
| `AGENT_WEB_SEARCH_API_KEY` | enables the `web_search` builtin tool (Tavily) |

## Tools

Deployed agents ship a **read-only** builtin tool set: `calculator`,
`current_datetime`, `fetch_url`, `http_request` (GET/HEAD only) and
`web_search` (when a key is configured). Mutating tools (e.g. `run_python`)
are deliberately not registered in deployments.

## Development

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

Status: P0 (see `PRODUCTION_TODO.md` in the care-workspace). The hub
(`/agents/<name>/…` multi-agent server with a control API) lands next.
