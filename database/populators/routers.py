"""
JSON-driven populator for the LLM provider routers.

Each entry of `routers.json` is a router *role* (`text`, `grounding`, ...). Agents name the role
they run on (`agents.json` → `router`), so they never depend on which model a role resolves to.

A role's provider, endpoint and key come from, in order:
1. `ROUTER_<ROLE>_PROVIDER` / `ROUTER_<ROLE>_API_BASE` / `ROUTER_<ROLE>_API_KEY` in the environment;
2. the entry's `provider_type` / `api_endpoint` / `api_key` (`$TOKEN`s are resolved from settings);
3. `PROVIDER` / `PROVIDER_API_BASE` / `PROVIDER_API_KEY`.

Roles that resolve to the same model on the same endpoint share one Router row, with the union of
their capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, select

from database.provider.models import Router
from settings import (
    FREE_PROVIDER_API_KEY,
    PROVIDER,
    PROVIDER_API_BASE,
    PROVIDER_API_KEY,
    PROVIDER_GROUNDING_MODEL,
    PROVIDER_GROUNDING_MODEL_BACKBONE,
    PROVIDER_MODEL,
    PROVIDER_VISION_MODEL,
    PROVIDER_VISION_TOOL_MODEL,
    ROUTER_OVERRIDES,
)

CONFIG_PATH = Path(__file__).parent / "routers.json"


@dataclass
class RouterSpec:
    """A router role from `routers.json`, with every token and override resolved."""

    roles: list[str]
    model_name: str
    api_endpoint: str
    api_key: str
    provider_type: Router.Provider
    capabilities: list[Router.Capability] = field(
        default_factory=lambda: [Router.Capability.TEXT]
    )
    vision_backbone: Router.VisionBackbone | None = None


def _load_json_config() -> dict[str, Any]:
    """
    Load router configuration from `database/populators/routers.json`.

    Expected structure:
    {
      "routers": [
        {
          "role": "grounding",  // unique; agents reference it, ROUTER_<ROLE>_* override it
          "model_name": "$PROVIDER_GROUNDING_MODEL" | "literal-model-name",
          "api_key": "$PROVIDER_API_KEY",  // optional, defaults to PROVIDER_API_KEY
          "api_endpoint": "$PROVIDER_API_BASE",  // optional, defaults to PROVIDER_API_BASE
          "provider_type": "VLLM",  // optional, defaults to PROVIDER
          "capabilities": ["TEXT", "VISION"],  // optional, defaults to ["TEXT"]
          "vision_backbone": "QWEN3.5"  // optional, defaults to OTHER for vision routers
        }
      ]
    }
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"[populate_routers] Missing JSON configuration file: {CONFIG_PATH}"
        )
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_token(value: str | None) -> str:
    """
    Resolve known tokens to values from settings, otherwise return the literal.
    """
    if value is None:
        return ""
    token_map = {
        "$FREE_PROVIDER_API_KEY": FREE_PROVIDER_API_KEY,
        "$PROVIDER_API_KEY": PROVIDER_API_KEY,
        "$PROVIDER_API_BASE": PROVIDER_API_BASE,
        "$PROVIDER_MODEL": PROVIDER_MODEL,
        "$PROVIDER_VISION_MODEL": PROVIDER_VISION_MODEL,
        "$PROVIDER_VISION_TOOL_MODEL": PROVIDER_VISION_TOOL_MODEL,
        "$PROVIDER_GROUNDING_MODEL": PROVIDER_GROUNDING_MODEL,
        "$PROVIDER_GROUNDING_MODEL_BACKBONE": PROVIDER_GROUNDING_MODEL_BACKBONE,
    }
    return token_map.get(value, value)


def _resolve_provider_type(name: str) -> Router.Provider:
    """Resolve a provider name (`openrouter`, `openai`, `vllm`), case-insensitively."""
    try:
        return Router.Provider(name.strip().lower())
    except ValueError as e:
        allowed = ", ".join(p.value for p in Router.Provider)
        raise ValueError(f"Unknown provider '{name}'. Use one of: {allowed}.") from e


def _resolve_capabilities(values: list[str] | None) -> list[Router.Capability]:
    """Resolve configured capability names, defaulting to text-only."""
    if not values:
        return [Router.Capability.TEXT]

    capabilities: list[Router.Capability] = []
    for value in values:
        try:
            capability = Router.Capability(value.upper())
        except ValueError as e:
            raise ValueError(f"Unknown router capability '{value}'.") from e
        if capability not in capabilities:
            capabilities.append(capability)
    return capabilities


def _resolve_vision_backbone(name: str | None) -> Router.VisionBackbone | None:
    """Resolve a vision backbone by value (`QWEN2.5`) or by name (`QWEN2_5`)."""
    if not name:
        return None
    normalized = name.strip().upper()
    for backbone in Router.VisionBackbone:
        if normalized in (backbone.value, backbone.name):
            return backbone
    allowed = ", ".join(b.value for b in Router.VisionBackbone)
    raise ValueError(f"Unknown vision backbone '{name}'. Use one of: {allowed}.")


def _resolve_spec(item: dict[str, Any]) -> RouterSpec:
    """Resolve one `routers.json` entry into a RouterSpec."""
    role = str(item["role"]).lower()
    overrides = ROUTER_OVERRIDES.get(role.upper(), {})

    model_name = _resolve_token(item["model_name"])
    if not model_name:
        raise ValueError(
            f"[populate_routers] Router role '{role}' has no model: set the variable "
            f"behind '{item['model_name']}' in .env."
        )

    provider_name = (
        overrides.get("PROVIDER")
        or _resolve_token(item.get("provider_type"))
        or PROVIDER
    )
    api_endpoint = (
        overrides.get("API_BASE")
        or _resolve_token(item.get("api_endpoint"))
        or PROVIDER_API_BASE
    )
    api_key = (
        overrides.get("API_KEY")
        or _resolve_token(item.get("api_key"))
        or PROVIDER_API_KEY
    )
    if not api_endpoint:
        raise ValueError(
            f"[populate_routers] Router role '{role}' has no endpoint: set PROVIDER_API_BASE "
            f"or ROUTER_{role.upper()}_API_BASE in .env."
        )

    return RouterSpec(
        roles=[role],
        model_name=model_name,
        api_endpoint=api_endpoint,
        api_key=api_key,
        provider_type=_resolve_provider_type(provider_name),
        capabilities=_resolve_capabilities(item.get("capabilities")),
        vision_backbone=_resolve_vision_backbone(
            _resolve_token(item.get("vision_backbone"))
        ),
    )


def _merge(into: RouterSpec, other: RouterSpec) -> None:
    """Fold a role that resolves to the same model and endpoint into an existing spec."""
    if (into.provider_type, into.api_key) != (other.provider_type, other.api_key):
        raise ValueError(
            f"[populate_routers] Router roles {into.roles} and {other.roles} use model "
            f"'{into.model_name}' at {into.api_endpoint} with different providers or keys."
        )
    into.roles.extend(other.roles)
    for capability in other.capabilities:
        if capability not in into.capabilities:
            into.capabilities.append(capability)
    into.vision_backbone = _merge_backbones(into, other.vision_backbone)


def _merge_backbones(
    spec: RouterSpec, other: Router.VisionBackbone | None
) -> Router.VisionBackbone | None:
    """Keep the specific backbone when one role says OTHER (or nothing) and the other names one."""
    generic = (None, Router.VisionBackbone.OTHER)
    if spec.vision_backbone in generic:
        return other if other is not None else spec.vision_backbone
    if other in generic or other == spec.vision_backbone:
        return spec.vision_backbone
    raise ValueError(
        f"[populate_routers] Router roles {spec.roles} share model '{spec.model_name}' "
        f"but disagree on its vision backbone ({spec.vision_backbone} vs {other})."
    )


def resolve_router_specs() -> dict[str, RouterSpec]:
    """Return every router role in `routers.json`, resolved and merged, keyed by role."""
    items: list[dict[str, Any]] = _load_json_config().get("routers", [])
    by_target: dict[tuple[str, str], RouterSpec] = {}
    by_role: dict[str, RouterSpec] = {}
    for item in items:
        spec = _resolve_spec(item)
        role = spec.roles[0]
        if role in by_role:
            raise ValueError(f"[populate_routers] Duplicate router role '{role}'.")
        key = (spec.model_name, spec.api_endpoint)
        if key in by_target:
            _merge(by_target[key], spec)
            spec = by_target[key]
        else:
            by_target[key] = spec
        by_role[role] = spec
    return by_role


def find_router(session: Session, role: str) -> Router:
    """Return the Router row seeded for a role, or raise if it doesn't exist yet."""
    specs = resolve_router_specs()
    spec = specs.get(role.lower())
    if spec is None:
        raise ValueError(
            f"Unknown router role '{role}'. Roles in routers.json: {', '.join(specs)}."
        )
    router = _existing_router(session, spec.model_name, spec.api_endpoint)
    if router is None:
        raise ValueError(
            f"No Router for role '{role}' (model '{spec.model_name}'). Run populate_routers first."
        )
    return router


def _existing_router(
    session: Session, model_name: str, api_endpoint: str
) -> Router | None:
    """
    Return an existing Router matching model_name + api_endpoint, or None.
    """
    return session.exec(
        select(Router).where(
            (Router.model_name == model_name) & (Router.api_endpoint == api_endpoint)
        )
    ).first()


def _create_router(session: Session, spec: RouterSpec) -> Router:
    """
    Create and persist a Router record.
    """
    router = Router(
        api_key=spec.api_key,
        model_name=spec.model_name,
        api_endpoint=spec.api_endpoint,
        provider_type=spec.provider_type,
        capabilities=list(spec.capabilities),
        vision_backbone=spec.vision_backbone,
    )
    session.add(router)
    session.commit()
    session.refresh(router)
    print(
        f"[populate_routers] Created {spec.provider_type.value} router for model "
        f"'{spec.model_name}' (roles: {', '.join(spec.roles)})."
    )
    return router


def _extend_router(session: Session, router: Router, spec: RouterSpec) -> bool:
    """Add capabilities a spec needs to an existing Router. Returns whether it changed."""
    missing = [c for c in spec.capabilities if not router.supports_capabilities(c)]
    if not missing:
        return False
    router.capabilities = [*router.capabilities, *missing]
    if router.vision_backbone in (None, Router.VisionBackbone.OTHER):
        router.vision_backbone = spec.vision_backbone
    router.normalize_vision_backbone()
    session.add(router)
    session.commit()
    print(
        f"[populate_routers] Added {', '.join(c.value for c in missing)} to router "
        f"'{router.model_name}' (roles: {', '.join(spec.roles)})."
    )
    return True


def populate_routers(engine: Engine) -> None:
    """
    Populate Router entries based on JSON configuration.
    An existing router for (model_name, api_endpoint) is kept, gaining any capability it lacks.
    """
    specs = resolve_router_specs()
    if not specs:
        print("[populate_routers] No routers found in JSON configuration.")
        return

    created = 0
    kept = 0
    unique_specs = list({id(spec): spec for spec in specs.values()}.values())
    with Session(engine) as session:
        for spec in unique_specs:
            existing = _existing_router(session, spec.model_name, spec.api_endpoint)
            if existing is None:
                _ = _create_router(session, spec)
                created += 1
            else:
                _ = _extend_router(session, existing, spec)
                kept += 1

    print(
        f"[populate_routers] Completed. Created: {created}, Kept (already existed): {kept}"
    )
