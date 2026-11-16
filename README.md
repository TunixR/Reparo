# R2Framework backend

R2Framework recovers failed RPA runs. When a software robot hits an exception, it opens a
WebSocket to this FastAPI backend and sends what it knows: the task, the UI log up to the failure,
the failed activity and the process variables. LLM agents built on
[Strands](https://strandsagents.com) analyse the failure and, when they can, take over the robot's
desktop through screenshots and UI actions to bring the process back to a state the robot can
continue from. The backend tells the robot where to resume, and records every agent, tool and GUI
step so a run can be inspected or exported afterwards.

## Quickstart

You need Docker with Compose v2, [uv](https://docs.astral.sh/uv/) and an OpenAI-compatible LLM
endpoint (OpenRouter, OpenAI, vLLM...). The full development stack uses about 2 GB of RAM.

1. Start Postgres, SeaweedFS (S3) and SigNoz (observability). Details: [`docker/README.md`](docker/README.md).

   ```bash
   cp docker/.env.example docker/.env   # fill in the passwords and the S3 keys
   docker compose -f docker/compose.dev.yaml up -d
   ```

   On a fresh install, create the SigNoz admin account once: open http://localhost:8080, or

   ```bash
   curl -X POST localhost:8080/api/v1/register -H 'Content-Type: application/json' \
     -d '{"name":"Admin","orgName":"r2framework","email":"admin@example.com","password":"<password>"}'
   ```

   Until it exists, `signoz-collector` stays unhealthy, `up --wait` fails and the otel-agent
   doesn't start. Then check that everything is healthy:

   ```bash
   docker compose -f docker/compose.dev.yaml up -d --wait
   ```

   Without observability, start only `postgres seaweedfs-s3` and set `OBSERVABILITY_ENABLED=false`
   in `.env`.

2. Configure the framework. Use the same Postgres and S3 credentials as `docker/.env`, set
   `SECRET_KEY` and `ADMIN_USERNAME`/`ADMIN_PASSWORD`, and fill in the `PROVIDER_*` models and key.
   Every router uses `PROVIDER` (`openrouter`, `openai` or `vllm`), `PROVIDER_API_BASE` and
   `PROVIDER_API_KEY`, unless you override them per router with `ROUTER_<ROLE>_*`. See
   [LLM providers](#llm-providers).

   ```bash
   cp .env.example .env
   ```

3. Install dependencies. Then apply the database migrations, create the S3 bucket, and seed the default agents,
   tools, provider routers and admin user. `populate.py` is safe to rerun.

   ```bash
   uv sync --all-extras --frozen
   uv run python populate.py
   ```

4. Run the API and open the API reference at http://localhost:8000/scalar.

   ```bash
   uv run fastapi dev main.py
   ```

5. Try a recovery. Log in as the admin and create a robot key:

   ```bash
   TOKEN=$(curl -s -X POST localhost:8000/auth/login \
     --data-urlencode "username=$ADMIN_USERNAME" --data-urlencode "password=$ADMIN_PASSWORD" \
     | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
   curl -s -X POST localhost:8000/keys -H "Authorization: Bearer $TOKEN" \
     -H 'Content-Type: application/json' -d '{"name":"my-robot"}'   # prints "key" once
   ```

   Then run a robot with that key:
   - [PlaywrightReparoBridge](https://github.com/TunixR/PlaywrightReparoBridge): a headless
     browser example. Run its spending-approval example with
     `--ui changed --reparo real --reparo-base-url http://localhost:8000`.
   - The desktop test client, see
     [`scripts/recovery_ws_client/README.md`](scripts/recovery_ws_client/README.md). It performs
     real mouse and keyboard actions on your desktop.

### LLM providers

`populate.py` seeds one router per role in
[`database/populators/routers.json`](database/populators/routers.json). Agents name the role they
run on (`agents.json` → `router`).

| Role | Model | Used by |
| --- | --- | --- |
| `text` | `PROVIDER_MODEL` | Gateway Orchestrator, UI Exception Handler |
| `grounding` | `PROVIDER_GROUNDING_MODEL`, backbone `PROVIDER_GROUNDING_MODEL_BACKBONE` | Tooled GuiAgent |
| `uitars` | `bytedance/ui-tars-1.5-7b` | UI Standalone UITARS (disabled) |
| `vision`, `vision_tool` | `PROVIDER_VISION_MODEL`, `PROVIDER_VISION_TOOL_MODEL` | Not used by the default agents |

Each `routers.json` entry names its settings directly: `"provider_type": "$ROUTER_<ROLE>_PROVIDER"`,
`"api_endpoint": "$ROUTER_<ROLE>_API_BASE"` and `"api_key": "$ROUTER_<ROLE>_API_KEY"`. A literal
value works too. In `settings.py` those settings default to `PROVIDER`, `PROVIDER_API_BASE` and
`PROVIDER_API_KEY` when unset. For example, to keep the text agents on
OpenRouter and run grounding on your own vLLM server:

```bash
PROVIDER=openrouter
PROVIDER_API_BASE=https://openrouter.ai/api/v1
ROUTER_GROUNDING_PROVIDER=vllm
ROUTER_GROUNDING_API_BASE=http://gpu-host:8000/v1
```

Roles that resolve to the same model on the same endpoint share one router, which gets both
roles' capabilities. Rerunning `populate.py` adds missing capabilities but doesn't change existing
routers. Change those through `/provider`.

## Repository map

| Path | What lives there |
| --- | --- |
| `main.py` | FastAPI app: routers, `/health`, `/scalar`, observability setup |
| `settings.py`, `config.py` | Environment settings (see `.env.example`) and fixed limits |
| `routers/` | HTTP and WebSocket endpoints: recovery, evals, agents, tools, provider, keys, auth, logging |
| `database/` | SQLModel models and engine, grouped by concept (agents, tools, provider, keys, auth, logging) |
| `database/populators/` | Seed data (`agents.json`, `routers.json`, default agent prompts in `default_prompts/`) and the code that loads it; run by `populate.py` |
| `gateway/` | Tools for the gateway agent (human hand-off is a placeholder) |
| `agent_tools/` | Strands tools and hooks shared by agents: screenshots, tool limits, trace logging |
| `robot/` | The recovery WebSocket from accept to close: keep-alive, frame routing, `abort` and disconnects |
| `templates/` | Pydantic models for recovery payloads and agent responses |
| `security/`, `middlewares/` | Password and robot-key hashing, JWT handling, auth dependencies |
| `s3/` | S3 client for screenshots |
| `observability/` | OpenTelemetry tracing, logging, redaction and domain spans (see [`OBSERVABILITY.md`](OBSERVABILITY.md)) |
| `evals/` | Offline evaluation runners and evaluators |
| `scripts/` | Developer tools, including the recovery WebSocket test client |
| `migrations/` | Alembic migrations |
| `docker/` | Compose stacks for development and production |
| `tests/` | `unit/` (SQLite, runs in CI) and `integration/` (needs the Docker stack) |
| `modules/` | Reserved for recovery modules; currently empty |

## Documentation

| Question | Read |
| --- | --- |
| What is this and how do I run it? | This file |
| How do the pieces fit together? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| What is implemented and what is planned? | [`REQUIREMENTS.md`](REQUIREMENTS.md) |
| How do I work in the repo (lint, tests, migrations, code style)? | [`AGENTS.md`](AGENTS.md) |
| How do I name commits, branches, issues and PRs? How do labels work? | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| Traces, logs and SigNoz | [`OBSERVABILITY.md`](OBSERVABILITY.md) |
| Docker services, ports, production | [`docker/README.md`](docker/README.md) |
| Endpoints and payload schemas | `/scalar` on a running server |
