"""move the legacy default recovery graph to the UI Recovery graph

Revision 0003 made the oldest gateway the one node of a "Default Recovery" graph, so
recoveries kept running the gateway agent alone, with its sub-agents as tools. The
seeded "UI Recovery" graph (``database/populators/agent_graphs.json``) was only created
by the populator, disabled, behind manual API calls. This revision makes those calls.

It only acts on a database where "Default Recovery" is still the enabled default
recovery graph and has a single entry node, the gateway. There it:

- gives the gateway its seeded description, prompt and ``GatewayDecision`` output
  model, unless it already has that model, and drops its sub-agent links: the UI
  recovery now runs as later nodes, not as the gateway's tools;
- creates the "Recovery Inferer" and "Continuation Inferer" agents that are missing,
  on the gateway's router, the Continuation Inferer with the required
  ``set_continuation`` tool (whose row is created if it is missing);
- enables the four node agents and disables "UI Exception Handler", which the graph
  replaces;
- disables "Default Recovery" and makes "UI Recovery" the enabled default recovery
  graph, creating it with its four nodes and three edges when it doesn't exist yet.

The GUI node is the agent named "Tooled GuiAgent"; without it nothing changes, as its
router can't be told here. The definitions are frozen below as they were when this
revision was written, so later edits to the seeds don't change what it writes.

The downgrade makes "Default Recovery" the default again, gives the gateway its legacy
prompt and ``ResponseToRPA`` output model, and gives the gateway back "UI Exception
Handler", enabled, as its sub-agent. "UI Recovery" is kept, disabled, and so are the
inferers' rows, which are seeded agents. It only acts when "UI Recovery" is the
default and "Default Recovery" exists.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-24 12:00:00.000000

"""

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LOG_PREFIX = "[migration 0008]"

LEGACY_GRAPH = "Default Recovery"
UI_RECOVERY_GRAPH = "UI Recovery"
UI_RECOVERY_MAX_NODE_EXECUTIONS = 4

GUI_AGENT = "Tooled GuiAgent"
RECOVERY_INFERER = "Recovery Inferer"
CONTINUATION_INFERER = "Continuation Inferer"
REPLACED_AGENT = "UI Exception Handler"
LEGACY_SUB_AGENT_LIMIT = 1

SET_CONTINUATION = "set_continuation"
SET_CONTINUATION_FN_MODULE = "agent_tools.utils.set_continuation"
SET_CONTINUATION_DESCRIPTION = (
    "State whether the recovery repaired the process and, if it did, where the RPA "
    "robot continues. This is the only way to name the continuation, and it must "
    "succeed before you finish.\n"
    "- recovered: false if the recovery failed and the process was not repaired; pass "
    "process_finished false, model_event_id null and loop_iteration []; they are "
    "ignored and the robot is told the recovery failed.\n"
    "- process_finished: true if the recovery completed the rest of the process, so "
    "the robot has nothing left to run; model_event_id and loop_iteration are then "
    "ignored.\n"
    "- model_event_id: the id of the task in `model` the robot runs next (a model "
    "event id, never a UI log event_id). Naming the errored task means 'retry it'; "
    "naming an earlier one means 'go back'.\n"
    "- loop_iteration: one 0-based pass per loop enclosing that task, outermost "
    "first; [] when the task is in no loop.\n"
    "If the continuation is invalid, the error says what is wrong and what is "
    "expected: correct it and call again."
)

LEGACY_GATEWAY_DESCRIPTION = "Central orchestrator that standardizes and routes external RPA error notifications."
LEGACY_GATEWAY_OUTPUT_MODEL: dict[str, Any] = {
    "name": "ResponseToRPA",
    "description": "The recovery outcome sent to the robot; only `success` is reported by the agent.",
    "fields": [
        {
            "name": "success",
            "type": "bool",
            "description": "Indicates whether the recovery was successful and the RPA can continue.",
        }
    ],
}

GATEWAY_DESCRIPTION = "Entry node of the recovery graph: classifies a robot's failure as a UI exception or escalates it to a human."

GATEWAY_PROMPT = """\
You are the Gateway Orchestrator of an RPA (robotic process automation) recovery framework. A robot reported a failed run. You classify the failure: either it is a UI exception, which the automated UI recovery can repair by acting on the robot's desktop, or it goes to a human operator.

## Inputs
You receive the robot's report as text:
- `task_name`: the business task the robot was running.
- `platform`, `os`: the RPA platform and the operating system of the robot.
- `variables`: process variables and known runtime values.
- `ui_log`: the robot's recent UI activities before the failure, oldest first.
- `errored_act`: the activity that failed, with its `error_code` and `error_description`.
- `model`: the process model the robot was following.

You only see text. The UI recovery that runs after you has its own tools, such as screenshots of the robot's desktop, so don't rule out a UI exception just because the report has no screenshot.

## Your tool
- `route_to_human`: escalates the failure to a human operator, with a short explanation of the failure and why it can't be recovered automatically.

## What to do
1. Classify the failure from `errored_act` (what the activity meant to do and its error), using `ui_log` for what happened right before it. Don't invent facts; if the evidence is ambiguous, go with the best-supported reading.
2. A UI exception is a failure in a user interface interaction or in the state of the user interface. Examples: an element that can't be found or clicked, an unexpected popup or dialog, a page that didn't load or changed layout, navigation that ended on the wrong screen, a field that couldn't be filled or read. Don't call any tool for it: the UI recovery runs after you.
3. Anything else goes to a human: call `route_to_human` once. Examples: a system or infrastructure error (crashed robot, lost connection, missing credentials or licence), a business rule violation, or invalid input data that no UI action can fix.

## Report
Report `ui_exception` true only for a UI exception, and false when you called `route_to_human`. In `rationale`, name the evidence from the report that decided it.
"""

GATEWAY_OUTPUT_MODEL: dict[str, Any] = {
    "name": "GatewayDecision",
    "description": "The gateway's classification of a failure, which routes the recovery.",
    "fields": [
        {
            "name": "ui_exception",
            "type": "bool",
            "description": "Whether the failure is a UI interaction or UI state "
            "failure that acting on the robot's desktop can recover.",
        },
        {
            "name": "rationale",
            "type": "str",
            "description": "The evidence from the robot's report behind the "
            "classification.",
        },
    ],
}

LEGACY_GATEWAY_PROMPT = """\
You are the Gateway Orchestrator of an RPA (robotic process automation) recovery framework. A robot reported a failed run. You decide who handles it: an automated recovery module, or a human operator.

## Inputs
You receive the robot's report as text:
- `task_name`: the business task the robot was running.
- `platform`, `os`: the RPA platform and the operating system of the robot.
- `variables`: process variables and known runtime values.
- `ui_log`: the robot's recent UI activities before the failure, oldest first.
- `errored_act`: the activity that failed, with its `error_code` and `error_description`.
- `model`: the process model the robot was following.

You only see text. The recovery modules have their own tools, such as screenshots of the robot's desktop, so don't route away from a module just because the report has no screenshot.

## Your tools
- `ui_exception_handler`: recovers failures in user interface interactions by acting on the robot's desktop. Examples: an element that can't be found or clicked, an unexpected popup or dialog, a page that didn't load or changed layout, navigation that ended on the wrong screen, a field that couldn't be filled or read.
- `route_to_human`: escalates the failure to a human operator, with a short explanation of the failure and why it can't be recovered automatically.

## What to do
1. Classify the failure from `errored_act` (what the activity meant to do and its error), using `ui_log` for what happened right before it. Don't invent facts; if the evidence is ambiguous, go with the best-supported reading.
2. If it is a UI interaction or UI state failure, call `ui_exception_handler` once, passing every input exactly as you received it.
3. Otherwise call `route_to_human`. Examples: a system or infrastructure error (crashed robot, lost connection, missing credentials or licence), a business rule violation, or invalid input data that no UI action can fix.
4. Always call one of the two. You can call `ui_exception_handler` only once: if it reports a failure, don't call it again, escalate with `route_to_human`.

## Report
Report `success` true only if `ui_exception_handler` was called and reported `success` true. Escalating to a human ends the automated recovery, so report `success` false then.
"""

RECOVERY_INFERER_DESCRIPTION = "Infers what a UI recovery must achieve from the robot's report: the kind of failure, where it happened in the process model and the task for the GUI agent."

RECOVERY_INFERER_PROMPT = """\
You are the Recovery Inferer of an RPA (robotic process automation) recovery framework. A robot failed while interacting with an application's user interface. You work out what the recovery must achieve and write it as a task for a GUI agent, which then carries it out on the robot's desktop.

You only read text. The GUI agent sees the screen and acts on it.

## Inputs
- `task_name`: the business task the robot was running.
- `platform`, `os`: the RPA platform and the operating system of the robot.
- `variables`: process variables and known runtime values.
- `ui_log`: the robot's recent UI activities before the failure, oldest first.
- `errored_act`: the activity that failed, with its error. This is the failure anchor.
- `model`: the process model, in BPMN XML or another textual format (e.g. an indented outline), whose task ids are model event ids.

How the UI log maps onto `model`:
- A model event is a task defined in `model`; its model event id is the task's `id`, the same on every run and loop pass.
- `event_id` (in `ui_log` and `errored_act`) only identifies an occurrence in this run, counting up. It is never a task of `model`.
- `model_event_id` (nullable, in `ui_log` and `errored_act`) is the id of the task in `model` that the entry performed. Use it to find the model event an entry performed.
- Only when `model_event_id` is missing, match `event_name` against the task names in `model`, ignoring any trailing ` #<n>` suffixes. If more than one task matches, say that the match is ambiguous.
- `event_name` ends with one ` #<pass>` per loop that encloses the event, outermost first and 0-based (e.g. `Click Approve #1#2` is the second pass of the outer loop and the third of the inner one). Read the current loop passes off the suffix of `errored_act.event_name`; no suffix means the event is not inside a loop.

## 1. Understand the failure
1. From `errored_act`, identify the failed interaction: what it meant to do (`action_type`, `application`, `input`) and how it failed (`error_code`, `error_description`).
2. From `ui_log`, reconstruct what happened right before it, looking for navigation drift, popups, stale pages or missing elements.
3. From `model`, `task_name` and `variables`, work out which part of the process the robot was in and where it was heading.
4. From the ` #<pass>` suffixes of `errored_act.event_name`, note whether the failure is inside a loop and in which pass. A last suffix of `#0` means the first pass of the innermost loop.

Use the inputs as the source of truth and don't invent facts. The original path already failed and may no longer exist, so plan for the goal, not for a replay of the failed steps.

## 2. Write the recovery task
The task is a short instruction for the GUI agent, describing the state the application must reach so the robot can resume. The failed activity is only one step of `task_name`: ask yourself what must happen next for the current business objective, not how to redo the failed action. Scope the task to the subtask of `model` the failure is in; also covering later steps is fine, stopping short of the subtask's goal is not.

Name the goal, not the means. More than one of these can apply (a form inside a loop, for example); apply all that do:
- Navigation: name the target screen ("Navigate to the invoice approval page"), not a menu, a sidebar, a button or making an item visible.
- Data entry or forms: forms change, so ask to complete the whole submission with the available data, without naming fields or values, and to stop with the goal not reached if the data can no longer be entered ("Complete the invoice submission using the available data, transforming it if needed. If it can't be entered, stop with the goal not reached"). Never say what to type: the GUI agent reads `variables` itself.
- Extraction: here, do name the information to obtain, taken from `variables`, and say to store it ("Obtain the invoice identifier and payment amount and store them in the matching variables"). Name what to obtain, never how (a tab, a copy, OCR).
- Loops: when `errored_act.event_name` ends in `#0`, the failure is in the first pass, and the same failure will likely repeat in every later pass. The task must then cover the current pass and every remaining pass, not only the current item, stopping at the first one that fails ("Process all pending invoices using the available data. If one can't be processed, stop with the goal not reached without continuing"), never "skip the ones that fail". In a later pass (`#1` or more), the current pass's data is the likely problem, so ask to complete the current pass only.
- When `model` offers alternative paths at the failure point (for example, an exclusive gateway), name the goal of each and let the GUI agent choose based on what it sees.

The task must end in a state where its outcome can be checked: a submitted form, a reached screen, a stored value. Typing data without submitting it, or opening a menu, doesn't recover anything. When forcing the goal could break a business rule or damage data, say to stop with the goal not reached instead.

The failed path may be what broke, so don't mention its UI elements or its values, and don't prescribe steps ("do X by doing Y"): the GUI agent may reach the goal in another way. If typing the password failed, write "Log in using the available credentials", not "Log in by entering the password".

| Good | Bad |
| --- | --- |
| Return to the dashboard | Open the sidebar; click the Dashboard button |
| Complete the employee onboarding using the available data; stop with the goal not reached if it can't be entered | Enter the start dates and durations |
| Obtain the order number and order status and store them in the matching variables | Open the Details tab and copy the values |

Before you report, check your draft and rewrite it until every answer is right:
1. Does it name a target state of `task_name`'s current objective?
2. Does it mention a field, button, identifier or value from the failed path or `variables` (other than the information an extraction must obtain)?
3. Does it end in a state that can be checked?
4. If `errored_act.event_name` ends in `#0`, does it cover every remaining pass of the loop, not only the current item?
5. If it involves a form, does it ask for the whole submission and say when to stop with the goal not reached?

Don't carry out the recovery yourself: the GUI agent runs your task after you.

## Report
- `failure_kind`: `navigation`, `data_entry` or `extraction` for the interaction that failed, else `other`.
- `failed_model_event_id`: the task id of `model` the failed activity performed, or null if it can't be told. Never a UI log `event_id`.
- `loop_passes`: the 0-based passes read off the suffix of `errored_act.event_name`, outermost first; `[]` when it has none.
- `recovery_task`: your final recovery task.
- `rationale`: why this task gets the process back on track, and which of the rules above applied.
"""

RECOVERY_INFERER_ARGUMENTS: list[dict[str, Any]] = [
    {
        "name": "task_name",
        "type": "str",
        "description": "Name of the task being recovered.",
    },
    {
        "name": "platform",
        "type": "str",
        "description": "Execution platform where the task runs.",
    },
    {
        "name": "os",
        "type": "str",
        "description": "Operating system where recovery is happening.",
    },
    {
        "name": "variables",
        "type": "dict",
        "description": "Runtime variables relevant to the task.",
    },
    {
        "name": "ui_log",
        "type": "list",
        "description": "Ordered UI execution history leading up to the failure.",
    },
    {
        "name": "errored_act",
        "type": "dict",
        "description": "Structured information about the action that failed.",
    },
    {
        "name": "model",
        "type": "str",
        "description": "Process model, in BPMN XML or another textual format, whose task ids "
        "are the model event ids that model_event_id refers to.",
    },
]

RECOVERY_INFERER_OUTPUT_MODEL: dict[str, Any] = {
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
            "description": "The task id of `model` the failed activity performed; "
            "null when it can't be told.",
        },
        {
            "name": "loop_passes",
            "type": "list[int]",
            "description": "0-based pass of each loop enclosing the failed activity, "
            "outermost first; [] when it is in no loop.",
        },
        {
            "name": "recovery_task",
            "type": "str",
            "description": "The instruction for the GUI agent: the state the "
            "application must reach.",
        },
        {
            "name": "rationale",
            "type": "str",
            "description": "Why this task gets the process back on track.",
        },
    ],
}

CONTINUATION_INFERER_DESCRIPTION = "Judges a GUI recovery against its task and tells the robot where to continue with set_continuation."

CONTINUATION_INFERER_PROMPT = """\
You are the Continuation Inferer of an RPA (robotic process automation) recovery framework, and an expert in business process models. A robot failed while interacting with an application's user interface, and a GUI agent then tried to recover it on the robot's desktop. You decide whether the recovery succeeded and tell the robot where to continue.

## Inputs
- `recovery_task`: the task the GUI agent was given.
- `gui_summary`: the GUI agent's own summary of what it did and the screen it ended on.
- `goal_reached`: whether the GUI agent reported reaching the goal within its action cap.
- `errored_act`: the activity that failed, with its `model_event_id` (the task of `model` it performed) when the robot sent it.
- `model`: the process model, in BPMN XML or another textual format (e.g. an indented outline), whose task ids are model event ids.

The recovery most likely doesn't cover the rest of the process and has left some tasks pending. It won't line up with `model` exactly either, since the original run had failed. Work out the state the recovery left the process in, for example after a form was submitted or once a certain screen was reached, and judge the outcome from what the GUI agent did and saw, not from its claim alone. With `goal_reached` false the recovery failed.

## Tell the robot the outcome
Always call `set_continuation`, whether the recovery succeeded or not. Its first argument, `recovered`, says whether the process was repaired:
- If the recovery failed, call it with `recovered` false (`set_continuation(recovered=false, process_finished=false, model_event_id=null, loop_iteration=[])`); the robot is told the process was not repaired, and you report `success` false.
- If it succeeded, call it with `recovered` true and where the robot continues. You can't report a successful recovery without it.

Where the robot continues, when `recovered` is true:
- with `process_finished` true if the recovery completed the rest of the process, so the robot has nothing left to run;
- otherwise pick the model event to run next from `model`, and call it with that task's id as `model_event_id`, exactly as `model` writes it (e.g. the `id` attribute of a BPMN task), and one 0-based pass per loop enclosing that task, outermost first (`[]` when it is in no loop).
- Read the current passes off `errored_act.event_name`: it ends with one ` #<pass>` per enclosing loop, outermost first. If the recovery moved on to the next pass of a loop, count it.
- Never pass a UI log `event_id` (`"8"`): it counts occurrences in this run and is never a task of `model`. Nor is a position in `model`.
- Naming the errored model event means "retry it"; naming an earlier one means "go back"; naming a later one means "the recovery already did everything before it".

If the tool rejects the continuation, correct it as its error says and call it again until it returns `ok`. Only `set_continuation` decides what the robot receives; your report can't change it.

Example. The process approves pending spending requests, one per loop pass:

```
process spending_approval
  open_requests "Open the requests page"
  for each_request: request_id in pending
    open_request "Open the request"
    if decide: amount within limit
      click_approve "Click Approve"
    otherwise
      click_reject "Click Reject"
    back_to_list "Return to the list"
```

The robot failed clicking Approve on the third request: `errored_act` has `model_event_id: "click_approve"`, `event_id: "8"` and `event_name: "Click Approve #2"`, so it is pass 2 of `each_request`. Depending on how the recovery went:
- couldn't click Approve, the robot should retry it: `set_continuation(recovered=true, process_finished=false, model_event_id="click_approve", loop_iteration=[2])`;
- clicked Approve, the robot continues after it: `model_event_id="back_to_list"`, `loop_iteration=[2]`;
- approved and went back to the list, the robot opens the next request: `model_event_id="open_request"`, `loop_iteration=[3]`;
- found the request must be rejected instead: `model_event_id="click_reject"`, `loop_iteration=[2]`;
- finished every remaining request itself: `set_continuation(recovered=true, process_finished=true, model_event_id=null, loop_iteration=[])`;
- couldn't recover: `set_continuation(recovered=false, process_finished=false, model_event_id=null, loop_iteration=[])`, and report `success` false.

`model_event_id="8"` (the UI log `event_id`) or `loop_iteration=[]` for `click_approve` would be rejected. With nested loops (outer over approvers, inner over their requests), the second approver's third request is `loop_iteration=[1, 2]`.

## Report
Your report's `success` agrees with the `recovered` you passed to `set_continuation`. Its `summary` says what the recovery did and where the robot continues.
"""

CONTINUATION_INFERER_ARGUMENTS: list[dict[str, Any]] = [
    {
        "name": "recovery_task",
        "type": "str",
        "description": "The task the GUI agent was given.",
    },
    {
        "name": "gui_summary",
        "type": "str",
        "description": "The GUI agent's summary of what it did and the screen it ended on.",
    },
    {
        "name": "goal_reached",
        "type": "bool",
        "description": "Whether the GUI agent reported reaching the goal within its action "
        "cap.",
    },
    {
        "name": "errored_act",
        "type": "dict",
        "description": "Structured information about the action that failed.",
    },
    {
        "name": "model",
        "type": "str",
        "description": "Process model, in BPMN XML or another textual format, whose task ids "
        "are the model event ids that model_event_id refers to.",
    },
]

CONTINUATION_INFERER_OUTPUT_MODEL: dict[str, Any] = {
    "name": "ContinuationReport",
    "description": "Whether a UI recovery succeeded, once `set_continuation` named the "
    "outcome.",
    "fields": [
        {
            "name": "success",
            "type": "bool",
            "description": "Whether the recovery succeeded; agrees with the "
            "`recovered` passed to `set_continuation`.",
        },
        {
            "name": "summary",
            "type": "str",
            "description": "What the recovery did and where the robot continues.",
        },
    ],
}

UI_RECOVERY_DESCRIPTION = "The gateway classifies the failure. A UI exception goes on to the Recovery Inferer, which writes the recovery task, the Tooled GuiAgent, which carries it out on the robot's desktop, and the Continuation Inferer, which tells the robot where to continue. Any other failure ends at the gateway, without success."


UI_RECOVERY_CONDITION: dict[str, Any] = {
    "field": "ui_exception",
    "op": "eq",
    "value": True,
}


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    legacy = _legacy_default(bind)
    if legacy is None:
        return
    legacy_graph_id, gateway_id = legacy
    gui_agent_id = _agent_id(bind, GUI_AGENT)
    if gui_agent_id is None:
        print(
            f"{LOG_PREFIX} No agent '{GUI_AGENT}'; '{LEGACY_GRAPH}' stays the default "
            "recovery graph."
        )
        return
    now = datetime.now(UTC)
    router_id = bind.execute(
        sa.text("SELECT router_id FROM agent WHERE id = :id"), {"id": gateway_id}
    ).scalar_one()

    _ = bind.execute(
        sa.text(
            "UPDATE agent SET description = :description, prompt = :prompt,"
            " output_model = CAST(:output_model AS jsonb), updated_at = :now"
            " WHERE id = :id"
            " AND (output_model ->> 'name') IS DISTINCT FROM :model_name"
        ),
        {
            "id": gateway_id,
            "description": GATEWAY_DESCRIPTION,
            "prompt": GATEWAY_PROMPT,
            "output_model": json.dumps(GATEWAY_OUTPUT_MODEL),
            "model_name": GATEWAY_OUTPUT_MODEL["name"],
            "now": now,
        },
    )
    _ = bind.execute(
        sa.text("DELETE FROM subagent WHERE parent_agent_id = :id"),
        {"id": gateway_id},
    )

    recovery_inferer_id = _ensure_agent(
        bind,
        RECOVERY_INFERER,
        RECOVERY_INFERER_DESCRIPTION,
        RECOVERY_INFERER_PROMPT,
        RECOVERY_INFERER_ARGUMENTS,
        RECOVERY_INFERER_OUTPUT_MODEL,
        router_id,
        now,
    )
    continuation_inferer_id = _ensure_agent(
        bind,
        CONTINUATION_INFERER,
        CONTINUATION_INFERER_DESCRIPTION,
        CONTINUATION_INFERER_PROMPT,
        CONTINUATION_INFERER_ARGUMENTS,
        CONTINUATION_INFERER_OUTPUT_MODEL,
        router_id,
        now,
    )
    _ensure_required_tool(
        bind, continuation_inferer_id, _set_continuation(bind, now), now
    )

    _ = bind.execute(
        sa.text(
            "UPDATE agent SET enabled = true, updated_at = :now"
            " WHERE id IN (:gateway, :recovery, :gui, :continuation) AND NOT enabled"
        ),
        {
            "gateway": gateway_id,
            "recovery": recovery_inferer_id,
            "gui": gui_agent_id,
            "continuation": continuation_inferer_id,
            "now": now,
        },
    )
    _ = bind.execute(
        sa.text(
            "UPDATE agent SET enabled = false, updated_at = :now"
            " WHERE name = :name AND enabled"
        ),
        {"name": REPLACED_AGENT, "now": now},
    )

    _ = bind.execute(
        sa.text(
            "UPDATE agentgraph SET enabled = false, is_default_recovery = false,"
            " updated_at = :now WHERE id = :id"
        ),
        {"id": legacy_graph_id, "now": now},
    )
    graph_id = _graph_id(bind, UI_RECOVERY_GRAPH)
    if graph_id is None:
        _create_ui_recovery_graph(
            bind,
            {
                "gateway": gateway_id,
                "recovery_inferer": recovery_inferer_id,
                "tooled_guiagent": gui_agent_id,
                "continuation_inferer": continuation_inferer_id,
            },
            now,
        )
    else:
        _ = bind.execute(
            sa.text(
                "UPDATE agentgraph SET enabled = true, is_default_recovery = true,"
                " updated_at = :now WHERE id = :id"
            ),
            {"id": graph_id, "now": now},
        )
    print(
        f"{LOG_PREFIX} '{UI_RECOVERY_GRAPH}' is the default recovery graph; "
        f"'{LEGACY_GRAPH}' is disabled."
    )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    graph_id = bind.execute(
        sa.text("SELECT id FROM agentgraph WHERE name = :name AND is_default_recovery"),
        {"name": UI_RECOVERY_GRAPH},
    ).scalar()
    legacy_graph_id = _graph_id(bind, LEGACY_GRAPH)
    if graph_id is None or legacy_graph_id is None:
        return
    gateway_id = bind.execute(
        sa.text(
            "SELECT agent_id FROM agentgraphnode"
            " WHERE agent_graph_id = :id AND is_entry"
        ),
        {"id": legacy_graph_id},
    ).scalar()
    if gateway_id is None:
        return
    now = datetime.now(UTC)
    _ = bind.execute(
        sa.text(
            "UPDATE agentgraph SET enabled = false, is_default_recovery = false,"
            " updated_at = :now WHERE id = :id"
        ),
        {"id": graph_id, "now": now},
    )
    _ = bind.execute(
        sa.text(
            "UPDATE agentgraph SET enabled = true, is_default_recovery = true,"
            " updated_at = :now WHERE id = :id"
        ),
        {"id": legacy_graph_id, "now": now},
    )
    _ = bind.execute(
        sa.text(
            "UPDATE agent SET description = :description, prompt = :prompt,"
            " output_model = CAST(:output_model AS jsonb), enabled = true,"
            " updated_at = :now WHERE id = :id"
            " AND (output_model ->> 'name') = :model_name"
        ),
        {
            "id": gateway_id,
            "description": LEGACY_GATEWAY_DESCRIPTION,
            "prompt": LEGACY_GATEWAY_PROMPT,
            "output_model": json.dumps(LEGACY_GATEWAY_OUTPUT_MODEL),
            "model_name": GATEWAY_OUTPUT_MODEL["name"],
            "now": now,
        },
    )
    handler_id = _agent_id(bind, REPLACED_AGENT)
    if handler_id is None:
        return
    _ = bind.execute(
        sa.text("UPDATE agent SET enabled = true, updated_at = :now WHERE id = :id"),
        {"id": handler_id, "now": now},
    )
    _ = bind.execute(
        sa.text(
            'INSERT INTO subagent (id, parent_agent_id, child_agent_id, "limit",'
            " created_at) SELECT :id, :parent, :child, :limit, :now"
            " WHERE NOT EXISTS (SELECT 1 FROM subagent"
            " WHERE parent_agent_id = :parent AND child_agent_id = :child)"
        ),
        {
            "id": uuid.uuid4(),
            "parent": gateway_id,
            "child": handler_id,
            "limit": LEGACY_SUB_AGENT_LIMIT,
            "now": now,
        },
    )


def _legacy_default(bind: Connection) -> tuple[uuid.UUID, uuid.UUID] | None:
    """The legacy graph and its gateway, while it is the enabled default with one node."""
    row = bind.execute(
        sa.text(
            "SELECT graph.id AS graph_id, node.agent_id FROM agentgraph AS graph"
            " JOIN agentgraphnode AS node ON node.agent_graph_id = graph.id"
            " WHERE graph.name = :name AND graph.enabled AND graph.is_default_recovery"
            " AND node.is_entry"
            " AND (SELECT count(*) FROM agentgraphnode"
            " WHERE agent_graph_id = graph.id) = 1"
        ),
        {"name": LEGACY_GRAPH},
    ).first()
    return None if row is None else (row.graph_id, row.agent_id)


def _agent_id(bind: Connection, name: str) -> uuid.UUID | None:
    return bind.execute(
        sa.text("SELECT id FROM agent WHERE name = :name ORDER BY created_at LIMIT 1"),
        {"name": name},
    ).scalar()


def _graph_id(bind: Connection, name: str) -> uuid.UUID | None:
    return bind.execute(
        sa.text("SELECT id FROM agentgraph WHERE name = :name"), {"name": name}
    ).scalar()


def _ensure_agent(
    bind: Connection,
    name: str,
    description: str,
    prompt: str,
    arguments: list[dict[str, Any]],
    output_model: dict[str, Any],
    router_id: uuid.UUID,
    now: datetime,
) -> uuid.UUID:
    """The id of the agent called ``name``, created as seeded when it is missing."""
    existing = _agent_id(bind, name)
    if existing is not None:
        return existing
    agent_id = uuid.uuid4()
    _ = bind.execute(
        sa.text(
            "INSERT INTO agent (id, name, description, prompt, input_type, enabled,"
            " router_id, created_at, updated_at, type, reasoning_effort, arguments,"
            " output_model) VALUES (:id, :name, :description, :prompt, 'TEXT', true,"
            " :router_id, :now, :now, 'Agent', 'MEDIUM', CAST(:arguments AS jsonb),"
            " CAST(:output_model AS jsonb))"
        ),
        {
            "id": agent_id,
            "name": name,
            "description": description,
            "prompt": prompt,
            "router_id": router_id,
            "now": now,
            "arguments": json.dumps(arguments),
            "output_model": json.dumps(output_model),
        },
    )
    print(f"{LOG_PREFIX} Created the agent '{name}'.")
    return agent_id


def _set_continuation(bind: Connection, now: datetime) -> uuid.UUID:
    """The id of the ``set_continuation`` tool, created when it is missing."""
    existing = bind.execute(
        sa.text("SELECT id FROM tool WHERE fn_module = :fn_module OR name = :name"),
        {"fn_module": SET_CONTINUATION_FN_MODULE, "name": SET_CONTINUATION},
    ).scalar()
    if existing is not None:
        return existing
    tool_id = uuid.uuid4()
    _ = bind.execute(
        sa.text(
            "INSERT INTO tool (id, name, description, fn_module, created_at,"
            " updated_at) VALUES (:id, :name, :description, :fn_module, :now, :now)"
        ),
        {
            "id": tool_id,
            "name": SET_CONTINUATION,
            "description": SET_CONTINUATION_DESCRIPTION,
            "fn_module": SET_CONTINUATION_FN_MODULE,
            "now": now,
        },
    )
    return tool_id


def _ensure_required_tool(
    bind: Connection, agent_id: uuid.UUID, tool_id: uuid.UUID, now: datetime
) -> None:
    """Give the agent the tool, required, unless it already has it."""
    _ = bind.execute(
        sa.text(
            'INSERT INTO agenttool (id, agent_id, tool_id, "limit", required,'
            " created_at) SELECT :id, :agent_id, :tool_id, NULL, true, :now"
            " WHERE NOT EXISTS (SELECT 1 FROM agenttool"
            " WHERE agent_id = :agent_id AND tool_id = :tool_id)"
        ),
        {"id": uuid.uuid4(), "agent_id": agent_id, "tool_id": tool_id, "now": now},
    )


def _create_ui_recovery_graph(
    bind: Connection, agents: dict[str, uuid.UUID], now: datetime
) -> None:
    """Create the enabled default "UI Recovery" graph over the node agents, by key."""
    graph_id = uuid.uuid4()
    _ = bind.execute(
        sa.text(
            "INSERT INTO agentgraph (id, name, description, enabled,"
            " is_default_recovery, max_node_executions, created_at, updated_at)"
            " VALUES (:id, :name, :description, true, true, :max_node_executions,"
            " :now, :now)"
        ),
        {
            "id": graph_id,
            "name": UI_RECOVERY_GRAPH,
            "description": UI_RECOVERY_DESCRIPTION,
            "max_node_executions": UI_RECOVERY_MAX_NODE_EXECUTIONS,
            "now": now,
        },
    )
    input_maps: dict[str, dict[str, str]] = {
        "tooled_guiagent": {"task": "recovery_inferer.recovery_task"},
        "continuation_inferer": {
            "gui_summary": "tooled_guiagent.summary",
            "goal_reached": "tooled_guiagent.goal_reached",
        },
    }
    nodes = {key: uuid.uuid4() for key in agents}
    for key, agent_id in agents.items():
        _ = bind.execute(
            sa.text(
                "INSERT INTO agentgraphnode (id, agent_graph_id, key, agent_id,"
                " is_entry, input_map) VALUES (:id, :graph_id, :key, :agent_id,"
                " :is_entry, CAST(:input_map AS jsonb))"
            ),
            {
                "id": nodes[key],
                "graph_id": graph_id,
                "key": key,
                "agent_id": agent_id,
                "is_entry": key == "gateway",
                "input_map": json.dumps(input_maps.get(key, {})),
            },
        )
    edges: list[tuple[str, str, dict[str, Any] | None]] = [
        ("gateway", "recovery_inferer", UI_RECOVERY_CONDITION),
        ("recovery_inferer", "tooled_guiagent", None),
        ("tooled_guiagent", "continuation_inferer", None),
    ]
    for source, target, condition in edges:
        _ = bind.execute(
            sa.text(
                "INSERT INTO agentgraphedge (id, agent_graph_id, source_node_id,"
                " target_node_id, priority, condition) VALUES (:id, :graph_id,"
                " :source, :target, 0, CAST(:condition AS jsonb))"
            ),
            {
                "id": uuid.uuid4(),
                "graph_id": graph_id,
                "source": nodes[source],
                "target": nodes[target],
                "condition": None if condition is None else json.dumps(condition),
            },
        )
    print(f"{LOG_PREFIX} Created the agent graph '{UI_RECOVERY_GRAPH}'.")
