"""Migration 0006 moves agent arguments and output models into ``agent`` as data.

Each test migrates a scratch database of its own on the configured server, so the
framework's database is never touched. The scratch database is dropped afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.engine import make_url

import settings
from database.agents.specs import FIELD_SPECS, OutputModelSpec
from database.general import ALEMBIC_INI

GATEWAY_ARGUMENTS = [
    ("task_name", "str", "The task name."),
    ("variables", "dict", "The variables."),
    ("ui_log", "list", "The UI log."),
    ("attempts", "int", "Attempts so far."),
]


@pytest.fixture
def scratch_url(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The URL of an empty database that ``alembic`` migrates instead of the real one."""
    name = f"reparo_migration_{uuid.uuid4().hex[:12]}"
    admin = pg_engine.execution_options(isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        _ = connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(settings.POSTGRES_URL).set(database=name)
    rendered = url.render_as_string(hide_password=False)
    # migrations/env.py reads the URL from settings each time a command runs.
    monkeypatch.setattr(settings, "POSTGRES_URL", rendered)
    try:
        yield rendered
    finally:
        with admin.connect() as connection:
            _ = connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture
def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


@contextmanager
def _connect(url: str) -> Iterator[Connection]:
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            yield connection
    finally:
        engine.dispose()


def _seed_0005(url: str, agents: dict[str, str | None]) -> dict[str, uuid.UUID]:
    """Write a router and one agent per name, with the given response models; the
    first agent also gets `GATEWAY_ARGUMENTS`."""
    now = datetime.now(UTC)
    ids = {name: uuid.uuid4() for name in agents}
    with _connect(url) as connection:
        router_id = uuid.uuid4()
        _ = connection.execute(
            text(
                "INSERT INTO router (id, api_key, model_name, api_endpoint, "
                "provider_type, capabilities, created_at, updated_at) VALUES "
                "(:id, '', 'm', 'http://x', 'OPENAI', '{}'::jsonb, :now, :now)"
            ),
            {"id": router_id, "now": now},
        )
        for name, response_model in agents.items():
            _ = connection.execute(
                text(
                    "INSERT INTO agent (id, name, description, prompt, response_model, "
                    "input_type, enabled, router_id, created_at, updated_at, type) "
                    "VALUES (:id, :name, 'd', 'p', :response_model, 'TEXT', true, "
                    ":router_id, :now, :now, 'Agent')"
                ),
                {
                    "id": ids[name],
                    "name": name,
                    "response_model": response_model,
                    "router_id": router_id,
                    "now": now,
                },
            )
        first = ids[next(iter(agents))]
        for argument, type_, description in GATEWAY_ARGUMENTS:
            _ = connection.execute(
                text(
                    "INSERT INTO argument (id, name, description, type, json_type, "
                    "agent_id) VALUES (:id, :name, :description, :type, 'x', :agent)"
                ),
                {
                    "id": uuid.uuid4(),
                    "name": argument,
                    "description": description,
                    "type": type_,
                    "agent": first,
                },
            )
    return ids


def _agents(url: str) -> dict[str, dict[str, Any]]:
    with _connect(url) as connection:
        rows = connection.execute(
            text("SELECT name, arguments, output_model FROM agent")
        ).all()
        return {
            row.name: {"arguments": row.arguments, "output_model": row.output_model}
            for row in rows
        }


def _arguments_table(url: str) -> list[tuple[str, str, str]]:
    with _connect(url) as connection:
        rows = connection.execute(
            text("SELECT name, type, description FROM argument ORDER BY ctid")
        ).all()
        return [(row.name, row.type, row.description) for row in rows]


def test_upgrade_moves_arguments_and_response_models_into_the_agent(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0005")
    _ = _seed_0005(
        scratch_url,
        {
            "Gateway": "templates.gateway.GatewayDecision",
            "Error": "templates.error_handler.RecoveryInference",
            "Plain": None,
        },
    )

    command.upgrade(alembic_config, "0006")

    agents = _agents(scratch_url)
    assert agents["Gateway"]["arguments"] == [
        {"name": name, "type": type_, "description": description}
        for name, type_, description in GATEWAY_ARGUMENTS
    ]
    assert agents["Plain"] == {"arguments": [], "output_model": None}
    gateway = OutputModelSpec.model_validate(agents["Gateway"]["output_model"])
    assert gateway.name == "GatewayDecision"
    assert [field.name for field in gateway.fields] == ["ui_exception", "rationale"]
    inference = OutputModelSpec.model_validate(agents["Error"]["output_model"])
    assert inference.name == "RecoveryInference"
    # Every converted definition is valid under today's rules.
    _ = FIELD_SPECS.validate_python(agents["Gateway"]["arguments"])


def test_upgrade_fails_on_a_response_model_it_cant_convert(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0005")
    _ = _seed_0005(scratch_url, {"Custom": "my_project.models.CustomReport"})

    with pytest.raises(RuntimeError, match=r"'Custom'.*my_project\.models"):
        command.upgrade(alembic_config, "0006")


def test_downgrade_rebuilds_the_argument_rows(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0005")
    _ = _seed_0005(scratch_url, {"Gateway": "templates.gateway.GatewayDecision"})
    command.upgrade(alembic_config, "0006")

    command.downgrade(alembic_config, "0005")

    assert _arguments_table(scratch_url) == GATEWAY_ARGUMENTS
    with _connect(scratch_url) as connection:
        response_model = connection.execute(
            text("SELECT response_model FROM agent")
        ).scalar_one()
        assert response_model is None

    command.upgrade(alembic_config, "head")
    assert _agents(scratch_url)["Gateway"]["arguments"] == [
        {"name": name, "type": type_, "description": description}
        for name, type_, description in GATEWAY_ARGUMENTS
    ]
