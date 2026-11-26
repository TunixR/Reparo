# Initializes the FastAPI application and includes the main entry point.
# It also sets up the database connection and includes the necessary routers.
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

from observability import (
    configure_observability,
    setup_strands_telemetry,
    shutdown_observability,
)
from routers.agent_graphs import router as agent_graphs_router
from routers.agents import router as agents_router
from routers.auth import router as auth_router
from routers.evals import router as evals_router
from routers.keys import router as keys_router
from routers.logging import router as logging_router
from routers.provider import router as provider_router
from routers.recovery import router as recovery_router
from routers.tools import router as tools_router
from templates.recovery_payload import RecoveryPayload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Set up Strands telemetry and flush observability data on shutdown."""
    setup_strands_telemetry()
    yield
    shutdown_observability()


app = FastAPI(lifespan=lifespan)

app.include_router(logging_router)
app.include_router(keys_router)

app.include_router(recovery_router)
app.include_router(evals_router)
app.include_router(agents_router)
app.include_router(agent_graphs_router)
app.include_router(tools_router)
app.include_router(provider_router)
app.include_router(auth_router)


def _openapi() -> dict[str, Any]:
    """The OpenAPI document, plus the recovery payload robots send first.

    WebSocket routes are not part of OpenAPI, so the payload model they take is
    added to the components by hand, where `/scalar` lists it among the models.
    """
    schema = FastAPI.openapi(app)
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    if RecoveryPayload.__name__ not in schemas:
        payload = RecoveryPayload.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        schemas.update(payload.pop("$defs", {}))
        schemas[RecoveryPayload.__name__] = payload
    return schema


app.openapi = _openapi


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/scalar", include_in_schema=False)
async def root():
    return get_scalar_api_reference(
        # Your OpenAPI document
        openapi_url=app.openapi_url,
        # Avoid CORS issues (optional)
        scalar_proxy_url="https://proxy.scalar.com",
    )


configure_observability(app)
