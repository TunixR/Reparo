"""define agent arguments and output models as data

An agent's arguments move from the ``argument`` table into ``agent.arguments``, a JSONB
list of field definitions (name, type, description), in the order the rows were
written. Their types are copied unchanged: the old ones (str, int, float, bool, list,
dict, None) are all valid type expressions.

``agent.response_model``, an import string, becomes ``agent.output_model``, the
definition of that class. The classes are frozen below as they were when this revision
was written, so the migration never imports ``templates``; an import string not listed
fails the upgrade. ``ResponseToRPA`` is listed for databases seeded before the agent
graphs, whose gateway still reports it; only its ``success`` field reached the LLM.

The downgrade rebuilds the ``argument`` rows, mapping each type back to the old
vocabulary as closely as it can, and leaves ``response_model`` NULL: an output model
defined as data has no class to name.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-20 18:00:00.000000

"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

FROZEN_OUTPUT_MODELS: dict[str, dict[str, Any]] = {
    "templates.gateway.GatewayDecision": {
        "name": "GatewayDecision",
        "description": "The gateway's classification of a failure, which routes the recovery.",
        "fields": [
            {
                "name": "ui_exception",
                "type": "bool",
                "description": "Whether the failure is a UI interaction or UI state failure that acting on the robot's desktop can recover.",
            },
            {
                "name": "rationale",
                "type": "str",
                "description": "The evidence from the robot's report behind the classification.",
            },
        ],
    },
    "templates.error_handler.RecoveryInference": {
        "name": "RecoveryInference",
        "description": "What a UI recovery must achieve, inferred from the robot's report.",
        "fields": [
            {
                "name": "failure_kind",
                "type": "Literal['navigation', 'data_entry', 'extraction', 'other']",
                "description": "The kind of interaction that failed.",
            },
            {
                "name": "failed_model_event_id",
                "type": "str | None",
                "description": "The task id of `model` the failed activity performed; null when it can't be told.",
            },
            {
                "name": "loop_passes",
                "type": "list[int]",
                "description": "0-based pass of each loop enclosing the failed activity, outermost first; [] when it is in no loop.",
            },
            {
                "name": "recovery_task",
                "type": "str",
                "description": "The instruction for the GUI agent: the state the application must reach.",
            },
            {
                "name": "rationale",
                "type": "str",
                "description": "Why this task gets the process back on track.",
            },
        ],
    },
    "templates.error_handler.UiExceptionReport": {
        "name": "UiExceptionReport",
        "description": None,
        "fields": [
            {
                "name": "result",
                "type": "str",
                "description": "Summary of the recovery execution result.",
            },
            {
                "name": "success",
                "type": "bool",
                "description": "Indicates if the recovery was successful.",
            },
            {
                "name": "process_finished",
                "type": "bool",
                "description": "Whether the recovery completed the rest of the process, as set with `set_continuation`.",
            },
            {
                "name": "continue_from_model_event_id",
                "type": "str | None",
                "description": "The model event id (a task id of `model`) the robot runs next, as set with `set_continuation`; null when the process is finished or the recovery failed.",
            },
            {
                "name": "loop_iteration",
                "type": "list[int]",
                "description": "0-based passes of the loops enclosing that task, outermost first, as set with `set_continuation`; [] when there is none.",
            },
        ],
    },
    "templates.error_handler.ContinuationReport": {
        "name": "ContinuationReport",
        "description": "Whether a UI recovery succeeded, once `set_continuation` named the outcome.",
        "fields": [
            {
                "name": "success",
                "type": "bool",
                "description": "Whether the recovery succeeded; agrees with the `recovered` passed to `set_continuation`.",
            },
            {
                "name": "summary",
                "type": "str",
                "description": "What the recovery did and where the robot continues.",
            },
        ],
    },
    "templates.gateway.ResponseToRPA": {
        "name": "ResponseToRPA",
        "description": "The recovery outcome sent to the robot; only `success` is reported by the agent.",
        "fields": [
            {
                "name": "success",
                "type": "bool",
                "description": "Indicates whether the recovery was successful and the RPA can continue.",
            }
        ],
    },
}


_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "None": "null",
}


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "agent",
        sa.Column(
            "arguments",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "agent",
        sa.Column(
            "output_model", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )
    bind = op.get_bind()
    arguments: dict[uuid.UUID, list[dict[str, str]]] = {}
    for row in bind.execute(
        sa.text("SELECT agent_id, name, type, description FROM argument ORDER BY ctid")
    ):
        arguments.setdefault(row.agent_id, []).append(
            {"name": row.name, "type": row.type, "description": row.description}
        )
    for agent_id, fields in arguments.items():
        _ = bind.execute(
            sa.text(
                "UPDATE agent SET arguments = CAST(:fields AS jsonb) WHERE id = :id"
            ),
            {"fields": json.dumps(fields), "id": agent_id},
        )
    for row in bind.execute(
        sa.text(
            "SELECT id, name, response_model FROM agent WHERE response_model IS NOT NULL"
        )
    ).all():
        definition = FROZEN_OUTPUT_MODELS.get(row.response_model)
        if definition is None:
            raise RuntimeError(
                f"Agent {row.name!r} ({row.id}) has the response model "
                f"{row.response_model!r}, which this migration can't convert; known: "
                f"{', '.join(FROZEN_OUTPUT_MODELS)}. Set it to one of them, or to NULL, "
                "and upgrade again."
            )
        _ = bind.execute(
            sa.text(
                "UPDATE agent SET output_model = CAST(:definition AS jsonb) WHERE id = :id"
            ),
            {"definition": json.dumps(definition), "id": row.id},
        )
    op.drop_table("argument")
    op.drop_column("agent", "response_model")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "agent",
        sa.Column("response_model", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.create_table(
        "argument",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("json_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["agent_id"], ["agent.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    bind = op.get_bind()
    for row in bind.execute(sa.text("SELECT id, arguments FROM agent")).all():
        for field in row.arguments:
            type_ = _legacy_type(field["type"])
            _ = bind.execute(
                sa.text(
                    "INSERT INTO argument "
                    "(id, name, description, type, json_type, agent_id) "
                    "VALUES (:id, :name, :description, :type, :json_type, :agent_id)"
                ),
                {
                    "id": uuid.uuid4(),
                    "name": field["name"],
                    "description": field["description"],
                    "type": type_,
                    "json_type": _JSON_TYPES[type_],
                    "agent_id": row.id,
                },
            )
    op.drop_column("agent", "output_model")
    op.drop_column("agent", "arguments")


def _legacy_type(expression: str) -> str:
    """The closest type the ``argument`` table knew: ``list[str]`` → ``list``,
    ``str | None`` → ``str``, a ``Literal`` → the type of its first value, ``Any`` →
    ``str``."""
    if expression in _JSON_TYPES:
        return expression
    members = [m.strip() for m in _top_level_union(expression) if m.strip() != "None"]
    first = members[0] if members else "None"
    if first.startswith(("list", "dict")):
        return first[:4]
    if first.startswith("Literal["):
        value = first.removeprefix("Literal[").lstrip()
        if re.match(r"-?\d+\.\d", value):
            return "float"
        if re.match(r"-?\d", value):
            return "int"
        if value.startswith(("True", "False")):
            return "bool"
        return "str"
    return first if first in _JSON_TYPES else "str"


def _top_level_union(expression: str) -> list[str]:
    """Split ``expression`` at the ``|`` outside brackets."""
    members: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expression):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "|" and depth == 0:
            members.append(expression[start:index])
            start = index + 1
    members.append(expression[start:])
    return members
