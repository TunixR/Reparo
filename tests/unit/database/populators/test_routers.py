"""The router populator: roles, per-role provider overrides, merging and agent lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

import settings
from database.populators import routers
from database.provider.models import Router

SETTINGS = {
    "PROVIDER": "openrouter",
    "PROVIDER_API_BASE": "https://openrouter.test/api/v1",
    "PROVIDER_API_KEY": "main-key",
    "FREE_PROVIDER_API_KEY": "",
    "PROVIDER_MODEL": "text-model",
    "PROVIDER_VISION_MODEL": "vision-model",
    "PROVIDER_VISION_TOOL_MODEL": "vision-model",
    "PROVIDER_GROUNDING_MODEL": "grounding-model",
    "PROVIDER_GROUNDING_MODEL_BACKBONE": "QWEN3.5",
}
ROLES = ("TEXT", "GROUNDING", "UITARS", "VISION", "VISION_TOOL")


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the populator at a routers.json, environment and settings of the test's choosing.

    The ROUTER_<ROLE>_* settings are derived from `env` the way settings.py derives them.
    """

    def _configure(
        entries: list[dict[str, Any]],
        env: dict[str, str] | None = None,
        **overrides: str,
    ) -> None:
        config = tmp_path / "routers.json"
        _ = config.write_text(json.dumps({"routers": entries}), encoding="utf-8")
        monkeypatch.setattr(routers, "CONFIG_PATH", config)
        values = {**SETTINGS, **overrides}
        for name, value in values.items():
            monkeypatch.setattr(settings, name, value)
        for role in ROLES:
            for setting in ("PROVIDER", "API_BASE", "API_KEY"):
                monkeypatch.delenv(f"ROUTER_{role}_{setting}", raising=False)
        for name, value in (env or {}).items():
            monkeypatch.setenv(name, value)
        key_default = values["PROVIDER_API_KEY"]
        free_default = values["FREE_PROVIDER_API_KEY"] or key_default
        for role in ROLES:
            defaults = {
                "PROVIDER": values["PROVIDER"],
                "API_BASE": values["PROVIDER_API_BASE"],
                "API_KEY": free_default if role.startswith("VISION") else key_default,
            }
            for setting, default in defaults.items():
                monkeypatch.setattr(
                    settings,
                    f"ROUTER_{role}_{setting}",
                    settings._router_setting(role, setting, default),  # pyright: ignore[reportPrivateUsage]
                )

    return _configure


def _shipped_entries() -> list[dict[str, Any]]:
    path = Path(routers.__file__).parent / "routers.json"
    return json.loads(path.read_text(encoding="utf-8"))["routers"]


def test_shipped_config_resolves_with_the_default_settings(configure):
    configure(_shipped_entries())

    specs = routers.resolve_router_specs()

    assert set(specs) == {"text", "vision", "vision_tool", "uitars", "grounding"}
    assert specs["grounding"].vision_backbone == Router.VisionBackbone.QWEN3_5
    assert specs["uitars"].vision_backbone == Router.VisionBackbone.QWEN2_5
    assert all(s.provider_type == Router.Provider.OPENROUTER for s in specs.values())


def test_an_empty_free_key_falls_back_to_the_provider_key(configure):
    configure(_shipped_entries())

    assert routers.resolve_router_specs()["vision"].api_key == "main-key"


@pytest.mark.parametrize("backbone", ["QWEN2.5", "QWEN2_5", "qwen2.5"])
def test_backbones_resolve_by_value_or_name(backbone: str):
    assert (
        routers._resolve_vision_backbone(backbone)  # pyright: ignore[reportPrivateUsage]
        == Router.VisionBackbone.QWEN2_5
    )


def test_an_unknown_backbone_names_the_allowed_values():
    with pytest.raises(ValueError, match=r"QWEN3\.5"):
        _ = routers._resolve_vision_backbone("QWEN3.8")  # pyright: ignore[reportPrivateUsage]


def test_a_role_override_sets_its_provider_endpoint_and_key(configure):
    configure(
        _shipped_entries(),
        env={
            "ROUTER_GROUNDING_PROVIDER": "vllm",
            "ROUTER_GROUNDING_API_BASE": "http://gpu.test:8000/v1",
            "ROUTER_GROUNDING_API_KEY": "",
        },
    )

    specs = routers.resolve_router_specs()

    grounding = specs["grounding"]
    assert grounding.provider_type == Router.Provider.VLLM
    assert grounding.api_endpoint == "http://gpu.test:8000/v1"
    assert grounding.api_key == "main-key"  # an empty override keeps the default
    assert specs["text"].provider_type == Router.Provider.OPENROUTER


def test_the_default_provider_applies_to_every_role(configure):
    configure(_shipped_entries(), PROVIDER="VLLM")

    specs = routers.resolve_router_specs()

    assert {s.provider_type for s in specs.values()} == {Router.Provider.VLLM}


def test_an_unknown_provider_names_the_allowed_values(configure):
    configure(_shipped_entries(), PROVIDER="anthropic")

    with pytest.raises(ValueError, match="openrouter, vllm"):
        _ = routers.resolve_router_specs()


def test_a_role_without_a_model_says_which_variable_to_set(configure):
    configure(_shipped_entries(), PROVIDER_GROUNDING_MODEL="")

    with pytest.raises(ValueError, match=r"no model_name.*\$PROVIDER_GROUNDING_MODEL"):
        _ = routers.resolve_router_specs()


def test_roles_on_the_same_model_merge_capabilities_and_keep_the_specific_backbone(
    configure,
):
    configure(_shipped_entries(), PROVIDER_GROUNDING_MODEL="text-model")

    specs = routers.resolve_router_specs()

    assert specs["text"] is specs["grounding"]
    assert specs["text"].capabilities == [
        Router.Capability.TEXT,
        Router.Capability.VISION,
    ]
    assert specs["text"].vision_backbone == Router.VisionBackbone.QWEN3_5


def test_roles_on_the_same_model_with_different_keys_are_rejected(configure):
    configure(
        _shipped_entries(),
        env={"ROUTER_GROUNDING_API_KEY": "other-key"},
        PROVIDER_GROUNDING_MODEL="text-model",
    )

    with pytest.raises(ValueError, match="different providers or keys"):
        _ = routers.resolve_router_specs()


def test_populate_creates_one_router_per_model_and_endpoint(
    configure, engine: Engine, session: Session
):
    configure(_shipped_entries(), PROVIDER_GROUNDING_MODEL="text-model")

    routers.populate_routers(engine)

    rows = session.exec(select(Router)).all()
    assert sorted(r.model_name for r in rows) == [
        "bytedance/ui-tars-1.5-7b",
        "text-model",
        "vision-model",
    ]
    assert routers.find_router(session, "grounding").supports_capabilities(
        Router.Capability.VISION
    )


def test_populate_rerun_adds_missing_capabilities_to_an_existing_router(
    configure, engine: Engine, session: Session
):
    configure(_shipped_entries())
    routers.populate_routers(engine)
    configure(_shipped_entries(), PROVIDER_GROUNDING_MODEL="text-model")

    routers.populate_routers(engine)

    session.expire_all()
    text = routers.find_router(session, "text")
    assert text.supports_capabilities(Router.Capability.VISION)
    assert text.vision_backbone == Router.VisionBackbone.QWEN3_5


def test_an_unknown_token_is_rejected(configure):
    configure([{**_shipped_entries()[0], "api_endpoint": "$NOT_A_SETTING"}])

    with pytest.raises(ValueError, match="Unknown setting token"):
        _ = routers.resolve_router_specs()


def test_literal_values_are_used_as_is(configure):
    configure(
        [
            {
                "role": "text",
                "model_name": "my-model",
                "provider_type": "vllm",
                "api_endpoint": "http://localhost:8001/v1",
                "api_key": "",
            }
        ]
    )

    spec = routers.resolve_router_specs()["text"]

    assert (spec.provider_type, spec.api_endpoint, spec.api_key) == (
        Router.Provider.VLLM,
        "http://localhost:8001/v1",
        "",
    )


def test_find_router_rejects_an_unknown_role(configure, session: Session):
    configure(_shipped_entries())

    with pytest.raises(ValueError, match="Unknown router role 'nope'"):
        _ = routers.find_router(session, "nope")
