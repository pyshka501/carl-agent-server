# carl-agent-server

Serve [CARL](https://github.com/Glazkoff/carl) reasoning chains as HTTP
**agents**. One chain = one agent: a FastAPI facade with `/invoke`, `/info`,
`/healthz` and its **own Swagger** (`/docs`) — the OpenAPI title, description
and version come from the chain's metadata, so the docs page reads as *that
agent's* documentation and users can try the agent right from `/docs`.

Part of the MMAR ecosystem: chains are authored in CARE / MAGE, stored &
versioned in **gigaevo Memory** (channels: `latest` → `stable`), executed by
[mmar-carl](https://pypi.org/project/mmar-carl/). In *attached* mode an agent
follows a Memory channel and **hot-reloads on `promote`/`pin`** — promote a
new version and the running agent picks it up with zero downtime; pin the old
version to roll back. A hot-reloaded version that fails preflight is rejected
and the healthy one keeps serving.

## Quickstart — the hub (recommended)

One lightweight process hosts N agents; each gets its own Swagger at
`/agents/<name>/docs`. Deploy/undeploy at runtime via the control API:

```bash
uv sync --group dev

AGENT_LLM_API_KEY=sk-... AGENT_LLM_MODEL=openai/gpt-4o \
uv run carl-agent-hub serve --port 8080

# deploy a chain JSON from disk (offline source)
curl -X POST localhost:8080/deployments \
     -H 'content-type: application/json' \
     -d '{"name": "demo", "chain_file": "./chain.json"}'

# deploy a chain entity from Memory, following its stable channel
curl -X POST localhost:8080/deployments \
     -H 'content-type: application/json' \
     -d '{"name": "weather", "entity_id": "<chain-uuid>", "channel": "stable"}'

open http://localhost:8080/agents/demo/docs   # try POST /invoke from Swagger
```

Deployments persist to `--state-file` (default `~/.care/agent-hub.json`) and
are restored on restart; `--no-persist` disables that.

### Hub control API

| Endpoint | Purpose |
|---|---|
| `GET /deployments` | list deployments (name, url, version, ready, runs) |
| `POST /deployments` | deploy (body = deployment spec; 409 duplicate, 422 unloadable) |
| `GET /deployments/{name}` | one deployment |
| `POST /deployments/{name}/reload` | re-fetch + preflight + swap now |
| `DELETE /deployments/{name}` | undeploy (unmounts the agent) |
| `GET /healthz` | hub liveness |

## Solo mode (one agent = one process)

```bash
AGENT_LLM_API_KEY=... AGENT_LLM_MODEL=openai/gpt-4o \
uv run carl-agent serve --chain-file ./chain.json --name demo --port 8001
# or attached: --entity-id <uuid> --channel stable
```

## Agent API (under `/agents/<name>` in the hub, or the root in solo mode)

| Endpoint | Purpose |
|---|---|
| `POST /invoke` | run the chain (`?mode=sync` default; `?mode=async` → 202 + run_id) |
| `POST /chat` | converse with the agent (`{message, session_id?}`); the dialogue so far is fed into the chain each turn — the chain is unchanged. Omit `session_id` to start a session (returned in the reply); sessions evict after the idle TTL |
| `GET /runs/{id}` | run status/result (answer, steps, tokens, time) |
| `GET /runs/{id}/events` | SSE step stream (replays history, ends with `result`) |
| `DELETE /runs/{id}` | cooperative cancel of a running run |
| `GET /info` | agent card (name, version, channel, required tools, readiness) |
| `GET /healthz` / `GET /readyz` | liveness / readiness (with the reason when 503) |
| `GET /docs` | this agent's own Swagger |

## Environment

| Variable | Purpose |
|---|---|
| `AGENT_LLM_API_KEY` / `AGENT_LLM_MODEL` / `AGENT_LLM_BASE_URL` | OpenAI-compatible LLM the chains run on |
| `AGENT_MEMORY_URL` / `AGENT_MEMORY_API_KEY` | gigaevo Memory (attached mode) |
| `AGENT_WEB_SEARCH_API_KEY` | enables the `web_search` builtin tool (Tavily) |

Per-deployment overrides (`llm_model`, `llm_api_key`, `memory_url`, …) exist on
the deployment spec, but prefer env vars: hub specs persist to the state file
verbatim, and secrets belong in the environment, not on disk.

## Auth

Set a per-agent `api_key` on the deployment (CARE's `/deploy` generates one)
and `/invoke`, `/chat`, `/runs/*` require it via `X-API-Key: <key>` (or
`Authorization: Bearer <key>`); `/healthz`, `/readyz`, `/info`, `/docs` stay
open. Loopback requests (127.0.0.1/::1) skip the check unless
`auth_allow_localhost=false`. No `api_key` set → auth is off (localhost demo).
Solo: `carl-agent serve --api-key <key>` (or `AGENT_API_KEY`). The hub's
state file holds these keys and is written `chmod 600`.

## Timeouts

Two layers bound a run. `chain_timeout_s` (default 300s) is the agent's hard
wall-clock deadline for the whole run. `step_timeout_s` (default 60s) is a
default per-step timeout injected at load into any step the author left
unbounded — capped never to exceed the chain-level `timeout`, so it tightens
but never loosens authored intent. Together a single hung step fails fast at
the step level instead of burning the whole run budget.

## Tools

Deployed agents ship a **read-only** builtin tool set: `calculator`,
`current_datetime`, `fetch_url`, `http_request` (GET/HEAD only — mutating
methods raise) and `web_search` (when a key is configured). Mutating tools
(e.g. `run_python`) are deliberately not registered in deployments.

## Development

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
uv run mypy src/
```

Status: Phase A of the production-mode plan is nearly complete (agent core,
async/SSE runs, attached hot-reload, the hub, CLIs). Next: run-records to
Memory, then the CARE control-plane integration (`/deploy` from the TUI).
