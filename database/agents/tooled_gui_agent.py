"""Tool-based GUI recovery agent.

This module implements a GUI recovery agent that interacts with the UI through
structured tool calls instead of parsing free-form text output. Each action
(click, type, scroll, etc.) is a Strands tool that dispatches the action over
WebSocket, waits for the client result, records a GUI trace, and returns a
fresh screenshot for the model to reason about.

It runs two ways on one core: as the ``tooled_guiagent`` tool of a parent agent
(and the eval socket), and as a graph node through ``run``, which returns a
``GuiRunReport``.

Telemetry goes through ``GuiTaskTelemetry``: the current span (the
``tooled_guiagent`` tool call or the graph node) carries
``gui.recovery_task.name`` and ``gui.task.*``, and each action tool span gets
``gui.action`` and ``gui.screenshot`` children (see OBSERVABILITY.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from database.agents.models import Agent

import asyncio
import html
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocketDisconnect
from pydantic import BaseModel, Field
from strands import Agent as StrandsAgent
from strands import ToolContext, tool
from strands.types.content import ContentBlock, Messages
from strands.types.tools import ToolResult
from strands.vended_plugins.context_injector import ContextInjector, InjectionContext

from agent_tools.context_compression import ReportingSummarizingManager
from agent_tools.hooks import AgentLoggingHook, MaxGuiImagesHook
from agent_tools.image import capture_screenshot
from observability.instrumentation.gui_agent import GuiAction, GuiTaskTelemetry
from robot.capabilities import require_vision
from settings import MAX_GUI_IMAGES

TOOL_NAME = "tooled_guiagent"
"""The tool name every GUI agent is exposed as to its parent agent."""

MAX_ACTIONS = 40

GUI_PRESERVED_MESSAGES = 2 * (MAX_GUI_IMAGES + 1)
"""Latest messages a summary never replaces: the last ``MAX_GUI_IMAGES + 1`` steps,
a tool call and its result each. They hold every screenshot still in the context
(``MaxGuiImagesHook`` keeps ``MAX_GUI_IMAGES`` and runs after the summary, so the
newest one is extra), and the summarizer reads only placeholders. A screenshot
costs one to three thousand tokens; describing old screens again buys nothing, as
the agent already wrote what it saw. ``note_progress`` and ``alter_variable`` steps
have no screenshot, so they can push one into the summarized range, where it is sent
to the vision model as it is."""

BUDGET_WARNING_RATIO = 0.75
"""From this share of ``MAX_ACTIONS`` on, the context block tells the model to wrap up."""

MAX_NOTE_CHARS = 300
"""Characters a progress note keeps; a longer one is cut."""

MAX_NOTES = 20
"""Progress notes a run keeps; it refuses more. Notes are for milestones, so a run
that reaches this many is looping, and the report stays short for its reader."""

NOTES_SHOWN = 5
"""The latest progress notes the context block lists; the report keeps them all."""

GUI_RUN_STATE_KEY = "gui_run"


def _coerce_to_held_type(held: Any, value: Any) -> Any:
    """``value`` with the type of ``held``, the Variable's current value.

    A Variable keeps the type the robot sent it with; one that holds a string or
    null accepts any value. A string is parsed as JSON first, and an int is
    accepted for a float. Any other type raises ``ValueError``.
    """
    if held is None or isinstance(held, str):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(held, float) and type(value) is int:
        value = float(value)
    if type(value) is not type(held):
        raise ValueError(f"expected {type(held).__name__}, got {json.dumps(value)}")
    return value


class GuiRunReport(BaseModel):
    """What a GUI agent run did on the robot's desktop, built in code."""

    goal_reached: bool = Field(
        description="The agent called finished with success and stayed within the action cap."
    )
    summary: str = Field(description="The agent's finished summary, or why it stopped.")
    actions: list[GuiAction] = Field(description="Actions sent to the robot, in order.")
    notes: list[str] = Field(
        description="The agent's progress notes (note_progress), oldest first."
    )


@dataclass(slots=True)
class _GuiRun:
    """One GUI run: what its tools record and what it ended with."""

    telemetry: GuiTaskTelemetry
    variables: dict[str, Any]
    agent: StrandsAgent
    conversation: ReportingSummarizingManager
    task: str
    actions: int = 0
    last_action: tuple[str, bool] | None = None
    """The tool name of the last action sent to the robot, and whether it succeeded."""
    cap_reached: bool = False
    notes: list[str] = field(default_factory=list)
    """Progress notes the model wrote with ``note_progress``, oldest first."""
    summary: str | None = None
    success: bool = False
    disconnect: WebSocketDisconnect | None = None
    error: str | None = None

    def context_block(self) -> str:
        """The run's state, as the model reads it before each call."""
        lines = [
            "<recovery_state>",
            # The task is model-written from robot data: escaped, it cannot close the block.
            f"<task>{html.escape(self.task, quote=False)}</task>",
            f"Actions used: {self.actions} of {MAX_ACTIONS}",
        ]
        if self.last_action is not None:
            name, succeeded = self.last_action
            result = "succeeded" if succeeded else "failed"
            lines.append(f"Last action: {name}, {result}.")
        if self.notes:
            shown = self.notes[-NOTES_SHOWN:]
            hidden = len(self.notes) > NOTES_SHOWN
            which = f" the last {NOTES_SHOWN} of {len(self.notes)}," if hidden else ""
            lines.append(f"Progress notes,{which} oldest first:")
            # Model-written too: escaped like the task.
            lines.extend(
                f"<note>{html.escape(note, quote=False)}</note>" for note in shown
            )
        left = MAX_ACTIONS - self.actions
        if left <= 0:
            lines.append(
                "Actions left: 0. Call finished now, with success=false"
                " if the goal is not reached."
            )
        elif self.actions >= BUDGET_WARNING_RATIO * MAX_ACTIONS:
            lines.append(
                f"Actions left: {left}. Finish the task, or call finished with"
                " success=false if the goal can't be reached."
            )
        lines.append("</recovery_state>")
        return "\n".join(lines)

    def report(self) -> GuiRunReport:
        if self.summary is not None:
            summary = self.summary
        elif self.cap_reached:
            summary = (
                f"Stopped at the cap of {MAX_ACTIONS} actions without calling finished."
            )
        else:
            summary = "Stopped without calling finished."
        return GuiRunReport(
            goal_reached=self.success and not self.cap_reached,
            summary=summary,
            actions=list(self.telemetry.action_log),
            notes=list(self.notes),
        )


class TooledGuiAgent:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.relative_dim = agent.router.relative_dim
        self.original_coords = agent.router.original_coords

    def _box_to_coords(self, box: list[float]) -> tuple[float, float]:
        if len(box) == 2:
            return box[0], box[1]
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    async def _execute_gui_action(
        self,
        action_type: str,
        action_inputs: dict[str, Any],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Dispatch one GUI action over WebSocket, record a trace, and return a fresh screenshot.

        Updates ``tool_context.invocation_state["current_image"]`` and
        ``tool_context.invocation_state["image_size"]`` in-place so subsequent
        tool calls see the latest screen dimensions.

        Raises:
            WebSocketDisconnect: when the client disconnects. Strands hands tool errors
                to the model, so the run also records it and stops the inner agent,
                and ``_run`` raises it again.
            RuntimeError: when the client reports action failure.
            RuntimeError: when the action limit is exceeded; the run then ends after
                the current tool calls.
        """
        state = tool_context.invocation_state
        run: _GuiRun = state[GUI_RUN_STATE_KEY]
        telemetry = run.telemetry
        if run.actions >= MAX_ACTIONS:
            if not run.cap_reached:
                run.cap_reached = True
                telemetry.max_actions_exceeded(telemetry.actions, MAX_ACTIONS)
            tool_context.agent.cancel(after_current_tools=True)
            raise RuntimeError(
                f"Action limit of {MAX_ACTIONS} reached: no more actions are sent and the run ends."
            )

        run.actions += 1
        succeeded = False
        try:
            result = await self._send_action(run, state, action_type, action_inputs)
            succeeded = True
            return result
        except WebSocketDisconnect as exc:
            run.disconnect = exc
            tool_context.agent.cancel()
            raise
        finally:
            run.last_action = (tool_context.tool_use["name"], succeeded)

    async def _send_action(
        self,
        run: _GuiRun,
        state: dict[str, Any],
        action_type: str,
        action_inputs: dict[str, Any],
    ) -> ToolResult:
        telemetry = run.telemetry
        websocket = state["websocket"]
        hook: AgentLoggingHook = state["hook"]
        current_image: bytes = state["current_image"]

        with telemetry.action(telemetry.actions + 1, action_type) as action_telemetry:
            started_at = datetime.now(UTC)
            await websocket.send_json(
                {
                    "type": "action",
                    "content": {
                        "action_type": action_type,
                        "action_inputs": action_inputs,
                    },
                }
            )
            result = await websocket.receive_json()
            finished_at = datetime.now(UTC)

            success = bool(result.get("success", False))
            action_telemetry.result(success)
            await hook.register_gui_trace(
                action_type,
                action_inputs,
                current_image,
                started_at,
                finished_at,
                success,
            )

            if not success:
                # The span gets a copy without the inputs, which may hold typed text.
                action_telemetry.failed(
                    RuntimeError(
                        f"Action execution failed on client side (action_type={action_type})."
                    )
                )
                # The model sees this message, so pass on the client's reason.
                raise RuntimeError(
                    f"Action '{action_type}' failed on client side. Inputs: {action_inputs}. "
                    f"Client error: {result.get('error') or 'unknown'}"
                )

        # Brief pause so the UI can settle before the next screenshot.
        await asyncio.sleep(1.0)

        new_image, image_size = await capture_screenshot(websocket, telemetry)
        state["current_image"] = new_image
        state["image_size"] = image_size

        match self.agent.router.model_name:
            case model if "Holo" in model:
                response = {
                    "status": "success",
                    "content": [
                        {"text": "<observation>\n"},
                        {"image": {"format": "jpeg", "source": {"bytes": new_image}}},
                        {"text": "\n<observation>"},
                    ],
                }
            case _:
                response = {
                    "status": "success",
                    "content": [
                        {"image": {"format": "jpeg", "source": {"bytes": new_image}}}
                    ],
                }

        return response  # pyright:  ignore[reportReturnType]

    # ---------------------------------------------------------------------------
    # GUI action tools
    # ---------------------------------------------------------------------------

    @tool(name="click", description="Left-click on a UI element.", context=True)
    async def click(
        self,
        start_box: list[float],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Left single click at the specified screen position.

        Args:
            start_box: Point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions.
        """
        image_size: tuple[int, int] = tool_context.invocation_state["image_size"]
        box = self.original_coords(
            *self._box_to_coords(start_box), image_size[0], image_size[1]
        )
        return await self._execute_gui_action(
            "left_single", {"start_box": box}, tool_context
        )

    @tool(
        name="double_click",
        description="Double left-click on a UI element.",
        context=True,
    )
    async def double_click(
        self,
        start_box: list[float],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Left double click at the specified screen position.

        Args:
            start_box: Point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions.
        """
        image_size: tuple[int, int] = tool_context.invocation_state["image_size"]
        box = self.original_coords(
            *self._box_to_coords(start_box), image_size[0], image_size[1]
        )
        return await self._execute_gui_action(
            "left_double", {"start_box": box}, tool_context
        )

    @tool(name="right_click", description="Right-click on a UI element.", context=True)
    async def right_click(
        self,
        start_box: list[float],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Right single click at the specified screen position.

        Args:
            start_box: Point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions.
        """
        image_size: tuple[int, int] = tool_context.invocation_state["image_size"]
        box = self.original_coords(
            *self._box_to_coords(start_box), image_size[0], image_size[1]
        )
        return await self._execute_gui_action(
            "right_single", {"start_box": box}, tool_context
        )

    @tool(
        name="hover",
        description="Move the mouse cursor over a UI element without clicking.",
        context=True,
    )
    async def hover(
        self,
        start_box: list[float],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Move the mouse to the specified screen position.

        Args:
            start_box: Point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions.
        """
        image_size: tuple[int, int] = tool_context.invocation_state["image_size"]
        box = self.original_coords(
            *self._box_to_coords(start_box), image_size[0], image_size[1]
        )
        return await self._execute_gui_action("hover", {"start_box": box}, tool_context)

    @tool(
        name="type_text",
        description="Type text into the currently focused input field.",
        context=True,
    )
    async def type_text(
        self,
        content: str,
        tool_context: ToolContext,
    ) -> ToolResult:
        """Type text into the currently active input field.

        Args:
            content: Text to type. Use \\n at the end to submit (press Enter).
        """
        return await self._execute_gui_action(
            "type", {"content": content}, tool_context
        )

    @tool(
        name="hotkey",
        description=(
            "Press a keyboard shortcut (e.g. 'ctrl c', 'alt f4', 'ctrl -' to zoom out, 'ctrl +' to zoom in). "
            "Split keys with a space, use lowercase, max 3 keys. Key names: ctrl, shift, alt, win, enter, "
            "esc, tab, space, backspace, delete, up, down, left, right, pageup, pagedown, home, end, f1-f12, "
            "or the character itself (a, 1, -, +, =)."
        ),
        context=True,
    )
    async def hotkey(
        self,
        key: str,
        tool_context: ToolContext,
    ) -> ToolResult:
        """Press a keyboard shortcut.

        Args:
            key: Space-separated key names in lowercase, e.g. 'ctrl v' or 'ctrl shift s'.
        """
        return await self._execute_gui_action("hotkey", {"hotkey": key}, tool_context)

    @tool(name="key_down", description="Press and hold a keyboard key.", context=True)
    async def key_down(
        self,
        key: str,
        tool_context: ToolContext,
    ) -> ToolResult:
        """Hold a key down (keydown event).

        Args:
            key: Key name, e.g. 'shift', 'ctrl', 'alt'.
        """
        return await self._execute_gui_action("keydown", {"key": key}, tool_context)

    @tool(
        name="key_up",
        description="Release a keyboard key that was held down.",
        context=True,
    )
    async def key_up(
        self,
        key: str,
        tool_context: ToolContext,
    ) -> ToolResult:
        """Release a held key (keyup event).

        Args:
            key: Key name, e.g. 'shift', 'ctrl', 'alt'.
        """
        return await self._execute_gui_action("keyup", {"key": key}, tool_context)

    @tool(
        name="scroll",
        description="Scroll in the given direction ('up', 'down', 'left', 'right'), optionally at a screen location.",
        context=True,
    )
    async def scroll(
        self,
        direction: str,
        start_box: list[float],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Scroll the mouse wheel.

        Args:
            direction: One of 'up', 'down', 'left', 'right'.
            start_box: Point [x, y] or box [x1, y1, x2, y2] inside the area to scroll
                (the page, a list, or a specific element), in the coordinate space given in the instructions.
        """
        action_inputs: dict[str, Any] = {"direction": direction}
        image_size: tuple[int, int] = tool_context.invocation_state["image_size"]
        action_inputs["start_box"] = self.original_coords(
            *self._box_to_coords(start_box), image_size[0], image_size[1]
        )
        return await self._execute_gui_action("scroll", action_inputs, tool_context)

    @tool(
        name="drag",
        description="Click and drag from one screen location to another.",
        context=True,
    )
    async def drag(
        self,
        start_box: list[float],
        end_box: list[float],
        tool_context: ToolContext,
    ) -> ToolResult:
        """Click and drag between two screen positions.

        Args:
            start_box: Drag origin as a point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions.
            end_box: Drag destination as a point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions.
        """
        image_size: tuple[int, int] = tool_context.invocation_state["image_size"]
        s_box = self.original_coords(
            *self._box_to_coords(start_box), image_size[0], image_size[1]
        )
        e_box = self.original_coords(
            *self._box_to_coords(end_box), image_size[0], image_size[1]
        )
        return await self._execute_gui_action(
            "drag", {"start_box": s_box, "end_box": e_box}, tool_context
        )

    @tool(
        name="wait",
        description="Wait for a specified number of seconds.",
        context=True,
    )
    async def wait(
        self,
        seconds: float,
        tool_context: ToolContext,
    ) -> ToolResult:
        """Wait/pause for the specified duration.

        Args:
            seconds: Number of seconds to wait.
        """
        return await self._execute_gui_action(
            "wait", {"seconds": seconds}, tool_context
        )

    @tool(
        name="finished",
        description="Signal that the recovery task is complete.",
        context=True,
    )
    async def finished(
        self,
        content: str,
        tool_context: ToolContext,
        success: bool = True,
    ) -> str:
        """Mark the task as complete and persist the final GUI trace.
        Call it when the recovery goal is reached, or when it clearly cannot be reached.

        Args:
            content: Summary of what was accomplished.
            success: Whether the recovery goal was reached; false when it cannot be.
        """
        state = tool_context.invocation_state
        run: _GuiRun = state[GUI_RUN_STATE_KEY]
        hook: AgentLoggingHook = state["hook"]
        current_image: bytes = state["current_image"]
        now = datetime.now(UTC)
        run.summary = content
        run.success = success
        if not run.cap_reached:
            run.telemetry.done(run.telemetry.actions)
        await hook.register_gui_trace(
            "finished", {"content": content}, current_image, now, now, success=True
        )
        tool_context.agent.cancel()
        return f"Task completed: {content}"

    @tool(
        name="note_progress",
        description=(
            "Keep a short progress note for the rest of the run. Your latest notes"
            " are shown to you on every step. It is not an action and doesn't use"
            " the action budget."
        ),
        context=True,
    )
    async def note_progress(self, note: str, tool_context: ToolContext) -> str:
        """Keep a short progress note for the rest of the run.

        Args:
            note: One short sentence, e.g. 'Invoice 42 found, total 17.50'.
        """
        run: _GuiRun = tool_context.invocation_state[GUI_RUN_STATE_KEY]
        # One line per note, so a note can't pass for another line of the block.
        note = " ".join(note.split())
        if not note:
            return "Error: the note is empty; nothing was noted."
        if len(run.notes) >= MAX_NOTES:
            return f"Error: the run keeps at most {MAX_NOTES} notes; nothing was noted."
        cut = len(note) > MAX_NOTE_CHARS
        run.notes.append(note[: MAX_NOTE_CHARS - 1] + "…" if cut else note)
        if cut:
            return f"Noted, cut to {MAX_NOTE_CHARS} characters; keep notes short."
        return "Noted."

    def get_gui_tools(self, variables: dict[str, Any]):
        return [
            self.click,
            self.double_click,
            self.right_click,
            self.hover,
            self.type_text,
            self.hotkey,
            self.key_down,
            self.key_up,
            self.scroll,
            self.drag,
            self.wait,
            self.finished,
            self.note_progress,
            self.create_alter_variable_tool(variables),
        ]

    def create_alter_variable_tool(self, variables: dict[str, Any]):
        @tool(
            name="alter_variable",
            description=(
                "Alter the value of a given variable. The value must be JSON"
                " serializable and keep the variable's JSON type."
            ),
        )
        async def _alter_variable(
            name: str,
            value: Any,
        ) -> str:
            if name not in variables:
                return f"Error: Variable '{name}' not found."
            variables[name] = _coerce_to_held_type(variables[name], value)
            return (
                f"Variable '{name}' has been updated to {json.dumps(variables[name])}."
            )

        return _alter_variable

    # ---------------------------------------------------------------------------
    # Outer tool consumed by the parent agent
    # ---------------------------------------------------------------------------

    @tool(
        name=TOOL_NAME,
        description="A element and action ground model for UI tasks.",
        context=True,
    )
    async def tooled_guiagent(
        self,
        task: str,
        variables: dict[str, Any],
        ui_log: list[Any],
        errored_act: dict[str, Any],
        model: str,
        override_invocation_state: dict[str, Any] | None = None,
        tool_context: ToolContext | None = None,
    ) -> tuple[list[list[ContentBlock]] | str, dict[str, Any]]:
        """Run the tool-based GUI recovery agent for a single recovery session.

        Args:
            task: Concise recovery objective and failure context.
            variables: Process variables and known runtime values.
            ui_log: Ordered UI execution history leading up to failure.
            errored_act: Structured details for the failed activity.
            model: The process model, whose task ids are model event ids.
            override_invocation_state: Optional override for the invocation state.

        Returns:
            Ordered list of assistant message content blocks from the inner agent,
            or a plain error string when an unexpected exception occurs.
        """

        state: dict[str, Any] = (
            {
                "inputs": {
                    "task": task,
                    "variables": variables,
                    "ui_log": ui_log,
                    "errored_act": errored_act,
                    "model": model,
                }
            }
            if tool_context is None
            else tool_context.invocation_state
        )
        if override_invocation_state is not None:
            state = override_invocation_state | dict(state)

        run = await self._run(
            state, task, variables, ui_log, errored_act, model, contain_errors=True
        )
        if run.error is not None:
            return run.error, run.variables
        # The summary, if any, stands in for the assistant turns it replaced.
        summary = run.conversation.summary_message
        conversation_history = [
            m["content"]
            for m in run.agent.messages
            if m["role"] == "assistant" or m is summary
        ]
        return conversation_history, run.variables

    async def run(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> GuiRunReport:
        """Drive the robot's desktop on bound inputs and report what happened.

        The direct run path of a graph node. Unlike the tool, it raises every error,
        so a robot disconnect or a model failure ends the node instead of a report.
        """
        run = await self._run(
            invocation_state,
            inputs["task"],
            inputs["variables"],
            inputs["ui_log"],
            inputs["errored_act"],
            inputs["model"],
            contain_errors=False,
        )
        return run.report()

    async def _run(
        self,
        state: dict[str, Any],
        task: str,
        variables: dict[str, Any],
        ui_log: list[Any],
        errored_act: dict[str, Any],
        model: str,
        *,
        contain_errors: bool,
    ) -> _GuiRun:
        """Run the inner agent until it finishes, stops or hits the action cap.

        A robot disconnect and a ``RuntimeError`` always propagate. Any other error
        propagates too, unless ``contain_errors``: then the run returns it as ``error``.
        """
        if "websocket" not in state:
            raise ValueError("WebSocket must be provided in tool context")
        require_vision(state)

        websocket = state["websocket"]
        state_variables = state.get("variables")
        if isinstance(state_variables, dict):
            variables = state_variables  # pyright: ignore[reportUnknownVariableType]
        # Enriches the current span: the Strands tool call or the graph node.
        telemetry = GuiTaskTelemetry(
            task=task, model=model, ui_log=ui_log, errored_act=errored_act
        )
        image, image_size = await capture_screenshot(websocket, telemetry)

        hook = AgentLoggingHook(
            agent_id=self.agent.id,
            invocation_state=state,
            parent_trace_id=state.get("parent_trace_id", None),
            is_gui_agent=True,
        )

        inner_state = dict(state) | {
            "hook": hook,
            "current_image": image,
            "image_size": image_size,
        }

        # Tools map coordinates back from this space (Router.original_coords).
        width, height = image_size
        model_height, model_width = self.relative_dim(height, width)
        instruction = f"""
    Task: {task}
    UI Log: {ui_log}
    Errored Activity: {errored_act}
    Variables: {variables}
    Model Context: {model}

    IMPORTANT: Give coordinates in a {model_width}x{model_height} space: x from 0 to {model_width} (left to right), y from 0 to {model_height} (top to bottom).
    """

        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {"text": self.agent.prompt.format(instruction=instruction)},
                    {"image": {"format": "jpeg", "source": {"bytes": image}}},
                ],
            }
        ]

        def render_state(_: InjectionContext) -> str:
            return run.context_block()

        # The task message is pinned: a summary never replaces it.
        conversation = ReportingSummarizingManager(
            pin_first=1,
            preserve_recent_messages=GUI_PRESERVED_MESSAGES,
            on_compression=hook.log_context_compression,
        )
        agent = StrandsAgent(  # type: ignore
            model=self.agent.get_model(),
            tools=self.get_gui_tools(variables),
            hooks=[hook, MaxGuiImagesHook()],
            conversation_manager=conversation,
            # Restates the task and the budget at the tail of every request, tool-result
            # turns included; only the latest block is sent.
            plugins=[ContextInjector(render_state, trigger="everyTurn")],
        )
        run = _GuiRun(
            telemetry=telemetry,
            variables=variables,
            agent=agent,
            conversation=conversation,
            task=task,
        )
        inner_state[GUI_RUN_STATE_KEY] = run

        try:
            _ = await run.agent.invoke_async(messages, invocation_state=inner_state)
            if run.disconnect is not None:
                raise run.disconnect
            if telemetry.outcome is None:
                telemetry.model_stopped(telemetry.actions)
            return run
        except WebSocketDisconnect:
            telemetry.client_disconnected(telemetry.actions)
            raise
        except RuntimeError as e:
            telemetry.failed(telemetry.actions, e, escaped=True)
            raise
        except Exception as e:
            telemetry.failed(telemetry.actions, e, escaped=not contain_errors)
            if not contain_errors:
                raise
            run.error = str(e)
            return run
        finally:
            hook.update_trace(finished=True, cost=-1.0)
