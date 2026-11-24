"""agent types as behaviours, starting with GUI agents

``agenttype`` becomes ``Agent``, ``TooledGuiAgent`` and ``TextGuiAgent``: how an agent
runs comes from its type's behaviour class (``database.agents.behaviours``).
Postgres can't drop an enum value, so the enum is replaced. Every ``GuiAgent``
becomes a ``TooledGuiAgent``, except the seeded "UI Standalone UITARS", which writes
UI-TARS actions as text and becomes a ``TextGuiAgent``: it gets the new
``parser_path`` column, and the text prompt, description and parser frozen below as
they were when this revision was written. Its old prompt was the tooled one, so
edits to it are replaced.

A GUI agent takes fixed inputs, so every GUI row gets the frozen GUI arguments and
the ``IMAGETEXT`` input type.

The seeded "UI Step Executor" is gone from the seeds: its links to tools and other
agents are deleted, then the row, unless an agent trace or a graph node still
references it; then it is disabled instead.

``subagent_not_self`` stops an agent from being its own sub-agent; rows that break it
are deleted first.

The downgrade maps both GUI types back to ``GuiAgent`` and drops the column and the
check. The prompts, arguments and the UI Step Executor are not restored.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-23 12:00:00.000000

"""

import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TYPES = ("Agent", "GuiAgent")
NEW_TYPES = ("Agent", "TooledGuiAgent", "TextGuiAgent")

UITARS = "UI Standalone UITARS"
UITARS_PARSER = "database.agents.behaviours.gui.parsers.uitars:UITarsParser"
UITARS_DESCRIPTION = "Recovers the UI by writing UI-TARS actions as text, which its parser reads and runs on the robot's desktop."
UI_STEP_EXECUTOR = "UI Step Executor"

GUI_ARGUMENTS: list[dict[str, Any]] = [
    {
        "name": "task",
        "type": "str",
        "description": "Task describing what needs to be done to get the process back on track.",
    },
    {"name": "variables", "type": "dict", "description": "Execution variables."},
    {
        "name": "ui_log",
        "type": "list",
        "description": "Ordered UI execution history leading up to the failure.",
    },
    {
        "name": "errored_act",
        "type": "dict | None",
        "description": "Structured information about the action that failed; null when the robot didn't name it.",
    },
    {
        "name": "model",
        "type": "str",
        "description": "Process model, in BPMN XML or another textual format, whose task ids are the model event ids that model_event_id refers to.",
    },
]

TEXT_GUI_PROMPT = """\
You are a GUI agent. You are given a task and your action history. You need to perform the next action to complete the task.
You will periodically be given screenshots to analyze the current UI state.

Furthermore, You are a specialized AI agent designed to recover robotic process automation (RPA) workflows that have failed.
You need to analyze the current state, understand what went wrong, and iteratively interact with the screen to get the process back on track.

Call context fields are simple and include:
1. `task`: concise recovery objective and failure context. YOU WILL NOT STOP UNTIL THIS TASK IS COMPLETED TO ITS FULLEST.
2. `variables`: process variables and known runtime values
3. `ui_log`: ordered recent UI execution history
4. `errored_act`: structured failed action details
5. `model`: the process model, in BPMN XML or another textual format, whose task ids are model event ids (`model_event_id` in `ui_log` and `errored_act`)

Use `errored_act` as the primary failure anchor and `ui_log` as the short-term execution timeline.

Use this grounding method before choosing actions:
1. Identify the failed interaction from `errored_act` and align it with the stated `task`.
2. Reconstruct immediate past context by reading the latest relevant `ui_log` entries.
3. Infer the intended continuation state using `variables` and `model` text.
4. Choose the next action that most reliably moves the UI toward that continuation state.

How to interpret context information:
- `errored_act.event_name` / `activity_name` (nullable), `action_type`, `application`, `input`, `error_code`, and `error_description` describe what failed and why.
- `errored_act.case_id` / `event_id` provide grouping and event context.
- `ui_log` entries describe what likely happened immediately before failure.
- `model` text provides additional continuation intent and constraints.

The inferred continuation state is a navigation hint: prioritize actions that move the UI toward a stable state where the robot can resume.
Do NOT treat prior actions as a fixed checklist to replay.
You must generate its own recovery path from current UI evidence.
It is allowed to choose an action that maps close to the original path when that is the best-supported option.
Always assume the original path may be invalid because it already failed.

Completion inference:
- You may use trace context to infer that recovery is complete when the current state appears to allow normal continuation of the remaining process.
- "Complete" means the robot can resume reliably from here, not that every original intermediate action was replayed exactly.

If evidence is ambiguous, prefer observable UI evidence and state the uncertainty in `Thought`.

Bear in mind that the UI error may have been caused by various factors, such as:
- Changes in the UI layout or elements
- Timing issues (elements not loading in time)
- Unexpected popups or dialogs
- Incorrect or missing input data

Correctly identifying the root cause is essential to effectively recover the process.

## Output Format
```
Previous action result: Success / Failure
Thought: ...
Action: ...
```

## Action Space

{actions}


## Note
- Use english in `Thought` part.
- On your first turn, you MUST reason in thought what the expected end result of the task is (e.g, reaching X screen, submitting a form, etc.) Based on the task description and model.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- Give coordinates in the space the instruction names.
- If the original task is completed, use the `finished` action to end the task. Use finished when and only when the task is fully completed. You must reason in thought about the task progress. Meaning, if task says complete X task, you cannot just do half of it and be done, even if the failure only covered the first half.
- Do not invent missing process facts; rely on screenshot evidence, `errored_act`, `ui_log`, `variables`, and `model`.

## User Instruction
{instruction}
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("agent", sa.Column("parser_path", sa.String(), nullable=True))
    _replace_enum(
        NEW_TYPES,
        f"""CASE
            WHEN type::text <> 'GuiAgent' THEN type::text
            WHEN name = '{UITARS}' THEN 'TextGuiAgent'
            ELSE 'TooledGuiAgent'
        END""",
    )
    agent = sa.table(
        "agent",
        sa.column("name", sa.String()),
        sa.column("type", sa.String()),
        sa.column("prompt", sa.String()),
        sa.column("description", sa.String()),
        sa.column("parser_path", sa.String()),
        sa.column("arguments", postgresql.JSONB()),
        sa.column("enabled", sa.Boolean()),
    )
    op.execute(
        agent.update()
        .where(agent.c.name == UITARS, sa.text("type = 'TextGuiAgent'"))
        .values(
            prompt=TEXT_GUI_PROMPT,
            description=UITARS_DESCRIPTION,
            parser_path=UITARS_PARSER,
        )
    )
    op.execute(
        sa.text(
            "UPDATE agent SET arguments = CAST(:arguments AS JSONB),"
            " input_type = 'IMAGETEXT'"
            " WHERE type IN ('TooledGuiAgent', 'TextGuiAgent')"
        ).bindparams(arguments=json.dumps(GUI_ARGUMENTS))
    )
    _delete_ui_step_executor()
    op.execute("DELETE FROM subagent WHERE parent_agent_id = child_agent_id")
    op.create_check_constraint(
        "subagent_not_self", "subagent", "parent_agent_id <> child_agent_id"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("subagent_not_self", "subagent", type_="check")
    _replace_enum(
        OLD_TYPES,
        "CASE WHEN type::text = 'Agent' THEN 'Agent' ELSE 'GuiAgent' END",
    )
    op.drop_column("agent", "parser_path")


def _delete_ui_step_executor() -> None:
    executor = "(SELECT id FROM agent WHERE name = :name)"
    for statement in (
        (
            f"DELETE FROM subagent WHERE parent_agent_id IN {executor}"
            f" OR child_agent_id IN {executor}"
        ),
        f"DELETE FROM agenttool WHERE agent_id IN {executor}",
        # Its traces and graph nodes keep the row; it is disabled instead.
        f"""DELETE FROM agent WHERE id IN {executor}
        AND NOT EXISTS (SELECT 1 FROM agenttrace WHERE agent_id = agent.id)
        AND NOT EXISTS (SELECT 1 FROM agentgraphnode WHERE agent_id = agent.id)""",
        "UPDATE agent SET enabled = false WHERE name = :name",
    ):
        op.execute(sa.text(statement).bindparams(name=UI_STEP_EXECUTOR))


def _replace_enum(values: Sequence[str], using: str) -> None:
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"CREATE TYPE agenttype_new AS ENUM ({labels})")
    op.execute(
        "ALTER TABLE agent ALTER COLUMN type TYPE agenttype_new"
        f" USING ({using})::agenttype_new"
    )
    op.execute("DROP TYPE agenttype")
    op.execute("ALTER TYPE agenttype_new RENAME TO agenttype")
