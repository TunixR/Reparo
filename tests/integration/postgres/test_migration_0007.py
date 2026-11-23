"""Migration 0007 turns ``GuiAgent`` into the GUI agent types and drops the UI Step Executor.

Each test migrates a scratch database of its own (``scratch_url``), so the framework's
database is never touched.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError

from database.agents.behaviours.gui.parsers import load_parser
from database.agents.behaviours.gui.shared import GUI_ARGUMENTS
from database.agents.specs import FIELD_SPECS, dump_fields
from database.populators.agents import DEFAULT_PROMPTS_DIR
from tests.integration.postgres.scratch import connect

UITARS = "UI Standalone UITARS"
TOOLED = "Tooled GuiAgent"
EXECUTOR = "UI Step Executor"
HANDLER = "UI Exception Handler"
OLD_ARGUMENTS = [{"name": "task", "type": "str", "description": "Task."}]


def _seed_0006(url: str) -> dict[str, uuid.UUID]:
    """The seeded GUI agents and the UI Step Executor as 0006 left them, all linked
    under the exception handler, the executor also to a tool."""
    now = datetime.now(UTC)
    types = {
        UITARS: "GuiAgent",
        TOOLED: "GuiAgent",
        EXECUTOR: "Agent",
        HANDLER: "Agent",
    }
    ids = {name: uuid.uuid4() for name in types}
    with connect(url) as connection:
        router_id = uuid.uuid4()
        _ = connection.execute(
            text(
                "INSERT INTO router (id, api_key, model_name, api_endpoint, "
                "provider_type, capabilities, created_at, updated_at) VALUES "
                "(:id, '', 'm', 'http://x', 'OPENAI', '{}'::jsonb, :now, :now)"
            ),
            {"id": router_id, "now": now},
        )
        for name, type_ in types.items():
            _ = connection.execute(
                text(
                    "INSERT INTO agent (id, name, description, prompt, input_type, "
                    "enabled, router_id, created_at, updated_at, type, arguments) "
                    "VALUES (:id, :name, 'd', 'tooled prompt', 'TEXT', true, "
                    ":router_id, :now, :now, :type, CAST(:arguments AS JSONB))"
                ),
                {
                    "id": ids[name],
                    "name": name,
                    "router_id": router_id,
                    "now": now,
                    "type": type_,
                    "arguments": '[{"name": "task", "type": "str", "description": "Task."}]',
                },
            )
        for child in (UITARS, TOOLED, EXECUTOR):
            _link(connection, ids[HANDLER], ids[child])
        tool_id = uuid.uuid4()
        _ = connection.execute(
            text(
                "INSERT INTO tool (id, name, description, fn_module, created_at, "
                "updated_at) VALUES (:id, 'ping', 'd', 'x.ping', :now, :now)"
            ),
            {"id": tool_id, "now": now},
        )
        _ = connection.execute(
            text(
                "INSERT INTO agenttool (id, agent_id, tool_id, required, created_at) "
                "VALUES (:id, :agent, :tool, false, :now)"
            ),
            {"id": uuid.uuid4(), "agent": ids[EXECUTOR], "tool": tool_id, "now": now},
        )
    return ids


def _link(connection: Connection, parent: uuid.UUID, child: uuid.UUID) -> None:
    _ = connection.execute(
        text(
            "INSERT INTO subagent (id, parent_agent_id, child_agent_id, created_at) "
            "VALUES (:id, :parent, :child, :now)"
        ),
        {
            "id": uuid.uuid4(),
            "parent": parent,
            "child": child,
            "now": datetime.now(UTC),
        },
    )


def _agents(url: str) -> dict[str, dict[str, Any]]:
    with connect(url) as connection:
        rows = connection.execute(text("SELECT * FROM agent")).mappings().all()
        return {row["name"]: dict(row) for row in rows}


def _children(url: str, parent: uuid.UUID) -> set[uuid.UUID]:
    with connect(url) as connection:
        return set(
            connection.execute(
                text("SELECT child_agent_id FROM subagent WHERE parent_agent_id = :p"),
                {"p": parent},
            ).scalars()
        )


def test_upgrade_gives_each_gui_agent_its_type(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0006")
    _ = _seed_0006(scratch_url)

    command.upgrade(alembic_config, "0007")

    agents = _agents(scratch_url)
    uitars, tooled = agents[UITARS], agents[TOOLED]
    assert (uitars["type"], tooled["type"], agents[HANDLER]["type"]) == (
        "TextGuiAgent",
        "TooledGuiAgent",
        "Agent",
    )
    # The UI-TARS agent writes text: its parser loads, and its prompt is the seed's.
    _ = load_parser(uitars["parser_path"])
    assert uitars["prompt"] == (DEFAULT_PROMPTS_DIR / "gui_agent_text.md").read_text(
        encoding="utf-8"
    )
    assert tooled["parser_path"] is None
    assert tooled["prompt"] == "tooled prompt"
    for gui in (uitars, tooled):
        assert gui["arguments"] == dump_fields(GUI_ARGUMENTS)
        assert gui["input_type"] == "IMAGETEXT"
        _ = FIELD_SPECS.validate_python(gui["arguments"])
    assert agents[HANDLER]["arguments"] == OLD_ARGUMENTS
    assert agents[HANDLER]["input_type"] == "TEXT"


def test_upgrade_deletes_the_ui_step_executor_and_its_links(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0006")
    ids = _seed_0006(scratch_url)

    command.upgrade(alembic_config, "0007")

    assert EXECUTOR not in _agents(scratch_url)
    assert _children(scratch_url, ids[HANDLER]) == {ids[UITARS], ids[TOOLED]}
    with connect(scratch_url) as connection:
        assert connection.execute(text("SELECT count(*) FROM agenttool")).scalar() == 0


def test_upgrade_disables_a_ui_step_executor_that_has_traces(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0006")
    ids = _seed_0006(scratch_url)
    with connect(scratch_url) as connection:
        _ = connection.execute(
            text(
                "INSERT INTO agenttrace (id, agent_id, inputs, cost, created_at) "
                "VALUES (:id, :agent, '{}'::jsonb, 0, :now)"
            ),
            {"id": uuid.uuid4(), "agent": ids[EXECUTOR], "now": datetime.now(UTC)},
        )

    command.upgrade(alembic_config, "0007")

    assert _agents(scratch_url)[EXECUTOR]["enabled"] is False
    assert _children(scratch_url, ids[HANDLER]) == {ids[UITARS], ids[TOOLED]}


def test_an_agent_cant_be_its_own_sub_agent(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0006")
    ids = _seed_0006(scratch_url)
    with connect(scratch_url) as connection:
        _link(connection, ids[HANDLER], ids[HANDLER])

    command.upgrade(alembic_config, "0007")

    # The self link 0006 allowed is gone, and a new one is rejected.
    assert ids[HANDLER] not in _children(scratch_url, ids[HANDLER])
    with pytest.raises(IntegrityError, match="subagent_not_self"):
        with connect(scratch_url) as connection:
            _link(connection, ids[HANDLER], ids[HANDLER])


def test_downgrade_maps_both_gui_types_back_to_gui_agent(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0006")
    ids = _seed_0006(scratch_url)
    command.upgrade(alembic_config, "0007")

    command.downgrade(alembic_config, "0006")

    agents = _agents(scratch_url)
    assert {agents[UITARS]["type"], agents[TOOLED]["type"]} == {"GuiAgent"}
    assert "parser_path" not in agents[UITARS]
    with connect(scratch_url) as connection:
        _link(connection, ids[HANDLER], ids[HANDLER])
        _ = connection.execute(
            text("DELETE FROM subagent WHERE parent_agent_id = child_agent_id")
        )

    command.upgrade(alembic_config, "head")
    assert _agents(scratch_url)[UITARS]["type"] == "TextGuiAgent"
