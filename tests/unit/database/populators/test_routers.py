"""The router populator: roles, per-role provider overrides, merging and agent lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

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


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the populator at a routers.json and settings of the test's choosing."""

    def _configure(
        entries: list[dict[str, Any]],
        overrides: dict[str, dict[str, str]] | None = None,
        **settings: str,
    ) -> None:
        config = tmp_path / "routers.json"
        _ = config.write_text(json.dumps({"routers": entries}), encoding="utf-8")
        monkeypatch.setattr(routers, "CONFIG_PATH", config)
        for name, value in {**SETTINGS, **settings}.items():
            monkeypatch.setattr(routers, name, value)
        monkeypatch.setattr(routers, "ROUTER_OVERRIDES", overrides or {})

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
    with pytest.raises(ValueError, match="QWEN3.5"):
        _ = routers._resolve_vision_backbone("QWEN3.8")  # pyright: ignore[reportPrivateUsage]


def test_a_role_override_sets_its_provider_endpoint_and_key(configure):
    configure(
        _shipped_entries(),
        overrides={
            "GROUNDING": {
                "PROVIDER": "vllm",
                "API_BASE": "http://gpu.test:8000/v1",
                "API_KEY": "",
            }
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

    with pytest.raises(ValueError, match=r"\$PROVIDER_GROUNDING_MODEL"):
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
        overrides={"GROUNDING": {"API_KEY": "other-key"}},
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


def test_find_router_rejects_an_unknown_role(configure, session: Session):
    configure(_shipped_entries())

    with pytest.raises(ValueError, match="Unknown router role 'nope'"):
        _ = routers.find_router(session, "nope")
