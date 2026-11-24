"""give the seeded agents and the UI Recovery graph the GUI progress notes

A GUI run's report now carries the agent's ``note_progress`` notes. The populators
never overwrite agents or graphs, so a seeded database gets them here:

- the "Continuation Inferer" gets the ``gui_notes`` argument, after ``goal_reached``,
  with a default of ``[]`` so other graphs that run it don't have to bind it, and a
  line describing it in its prompt;
- the "continuation_inferer" node of the "UI Recovery" graph maps ``gui_notes`` to the
  notes of the GUI node its ``gui_summary`` comes from;
- the "Tooled GuiAgent" and "UI Standalone UITARS" prompts get a line on when to
  take a note, and the UI-TARS prompt says to finish with success=False when the task
  can't be completed.

Each prompt line goes in after a line of the seeded prompt, and only when that line is
still there and the new one isn't, so an edited prompt is left alone. The texts are
frozen below as they were when this revision was written.

The downgrade removes what the upgrade added, where it is still as written.

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-24 18:00:00.000000

"""

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LOG_PREFIX = "[migration 0010]"

CONTINUATION_INFERER = "Continuation Inferer"
TOOLED_GUI_AGENT = "Tooled GuiAgent"
UITARS = "UI Standalone UITARS"
UI_RECOVERY_GRAPH = "UI Recovery"
CONTINUATION_NODE = "continuation_inferer"

GUI_NOTES = "gui_notes"
GUI_NOTES_AFTER = "goal_reached"
GUI_NOTES_ARGUMENT: dict[str, Any] = {
    "name": GUI_NOTES,
    "type": "list[str]",
    "description": "The progress notes the GUI agent kept during the run, oldest first.",
    "default": [],
}

NOTE_LINE = (
    "- `note_progress`: note a milestone in one short sentence: a subgoal done, a key "
    "value found, or a change of approach. Only note milestones, not every action: "
    "each call costs time and tokens."
)

PROMPT_INSERTS: dict[str, list[tuple[str, str]]] = {
    CONTINUATION_INFERER: [
        (
            (
                "- `goal_reached`: whether the GUI agent reported reaching the goal within "
                "its action cap."
            ),
            (
                "- `gui_notes`: the progress notes the GUI agent kept along the way, oldest "
                "first, such as a subgoal it finished or a value it found; often empty."
            ),
        )
    ],
    TOOLED_GUI_AGENT: [
        (
            (
                "- `alter_variable`: store a value the robot needs later, or one the task "
                "asks you to retrieve."
            ),
            NOTE_LINE,
        )
    ],
    UITARS: [
        (
            (
                "- Do not invent missing process facts; rely on screenshot evidence, "
                "`errored_act`, `ui_log`, `variables`, and `model`."
            ),
            NOTE_LINE,
        )
    ],
}
"""Per agent, each line to add and the seeded line it goes after."""

UITARS_FINISHED = "Use finished when and only when the task is fully completed."
UITARS_FINISHED_NOW = (
    "Use finished with success=True when and only when the task is fully completed, "
    "and with success=False when it clearly can't be completed."
)


def upgrade() -> None:
    bind = op.get_bind()
    now = datetime.now(UTC)
    _add_gui_notes_argument(bind, now)
    _map_gui_notes(bind)
    for name, inserts in PROMPT_INSERTS.items():
        _edit_prompt(bind, name, now, partial(_insert_lines, inserts=inserts))
    _edit_prompt(
        bind,
        UITARS,
        now,
        lambda p: p.replace(UITARS_FINISHED, UITARS_FINISHED_NOW, 1),
    )


def downgrade() -> None:
    bind = op.get_bind()
    now = datetime.now(UTC)
    _drop_gui_notes_argument(bind, now)
    _unmap_gui_notes(bind)
    for name, inserts in PROMPT_INSERTS.items():
        _edit_prompt(bind, name, now, partial(_remove_lines, inserts=inserts))
    _edit_prompt(
        bind,
        UITARS,
        now,
        lambda p: p.replace(UITARS_FINISHED_NOW, UITARS_FINISHED, 1),
    )


def _agent(bind: Connection, name: str) -> Any:
    return bind.execute(
        sa.text(
            "SELECT id, prompt, arguments FROM agent WHERE name = :name"
            " ORDER BY created_at LIMIT 1"
        ),
        {"name": name},
    ).first()


def _add_gui_notes_argument(bind: Connection, now: datetime) -> None:
    agent = _agent(bind, CONTINUATION_INFERER)
    if agent is None:
        return
    arguments: list[dict[str, Any]] = list(agent.arguments)
    names = [argument["name"] for argument in arguments]
    if GUI_NOTES in names:
        return
    position = (
        names.index(GUI_NOTES_AFTER) + 1 if GUI_NOTES_AFTER in names else len(arguments)
    )
    arguments.insert(position, GUI_NOTES_ARGUMENT)
    _set_arguments(bind, agent.id, arguments, now)
    print(f"{LOG_PREFIX} '{CONTINUATION_INFERER}' takes '{GUI_NOTES}'.")


def _drop_gui_notes_argument(bind: Connection, now: datetime) -> None:
    agent = _agent(bind, CONTINUATION_INFERER)
    if agent is None:
        return
    arguments: list[dict[str, Any]] = list(agent.arguments)
    kept = [argument for argument in arguments if argument != GUI_NOTES_ARGUMENT]
    if len(kept) != len(arguments):
        _set_arguments(bind, agent.id, kept, now)


def _set_arguments(
    bind: Connection, agent_id: Any, arguments: list[dict[str, Any]], now: datetime
) -> None:
    _ = bind.execute(
        sa.text(
            "UPDATE agent SET arguments = CAST(:arguments AS jsonb), updated_at = :now"
            " WHERE id = :id"
        ),
        {"id": agent_id, "arguments": json.dumps(arguments), "now": now},
    )


def _continuation_node(bind: Connection) -> Any:
    return bind.execute(
        sa.text(
            "SELECT node.id, node.input_map FROM agentgraphnode AS node"
            " JOIN agentgraph AS graph ON graph.id = node.agent_graph_id"
            " WHERE graph.name = :graph AND node.key = :key"
        ),
        {"graph": UI_RECOVERY_GRAPH, "key": CONTINUATION_NODE},
    ).first()


def _map_gui_notes(bind: Connection) -> None:
    node = _continuation_node(bind)
    if node is None:
        return
    input_map: dict[str, str] = dict(node.input_map)
    source, _, field = input_map.get("gui_summary", "").rpartition(".")
    if GUI_NOTES in input_map or not source or field != "summary":
        return
    input_map[GUI_NOTES] = f"{source}.notes"
    _set_input_map(bind, node.id, input_map)
    print(f"{LOG_PREFIX} '{UI_RECOVERY_GRAPH}' maps '{GUI_NOTES}' to '{source}.notes'.")


def _unmap_gui_notes(bind: Connection) -> None:
    node = _continuation_node(bind)
    if node is None:
        return
    input_map: dict[str, str] = dict(node.input_map)
    if not input_map.get(GUI_NOTES, "").endswith(".notes"):
        return
    del input_map[GUI_NOTES]
    _set_input_map(bind, node.id, input_map)


def _set_input_map(bind: Connection, node_id: Any, input_map: dict[str, str]) -> None:
    _ = bind.execute(
        sa.text(
            "UPDATE agentgraphnode SET input_map = CAST(:input_map AS jsonb)"
            " WHERE id = :id"
        ),
        {"id": node_id, "input_map": json.dumps(input_map)},
    )


def _edit_prompt(
    bind: Connection, name: str, now: datetime, edit: Callable[[str], str]
) -> None:
    agent = _agent(bind, name)
    if agent is None:
        return
    prompt = edit(agent.prompt)
    if prompt == agent.prompt:
        return
    _ = bind.execute(
        sa.text("UPDATE agent SET prompt = :prompt, updated_at = :now WHERE id = :id"),
        {"id": agent.id, "prompt": prompt, "now": now},
    )


def _insert_lines(prompt: str, inserts: list[tuple[str, str]]) -> str:
    lines = prompt.split("\n")
    for after, line in inserts:
        if after in lines and line not in lines:
            lines.insert(lines.index(after) + 1, line)
    return "\n".join(lines)


def _remove_lines(prompt: str, inserts: list[tuple[str, str]]) -> str:
    lines = prompt.split("\n")
    for _, line in inserts:
        if line in lines:
            lines.remove(line)
    return "\n".join(lines)
