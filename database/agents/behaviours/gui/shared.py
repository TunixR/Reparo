"""What both GUI agent loops share: their inputs and report, the actions, and a run.

Each action is defined once, in `DEFAULT_GUI_ACTIONS`: its name, description,
inputs, and how they reach the robot as an ``action`` message (coordinates
mapped back to the screen). `TooledGuiAgent` turns each one into a Strands
tool; `TextGuiAgent` describes them to its model and runs what its parser reads.
Both run them through `GuiRun.perform`, which sends the action, records its GUI
trace and ``gui.action`` span, and returns the next observation.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import WebSocketDisconnect
from pydantic import AfterValidator, BaseModel, Field, ValidationError
from strands import Agent as StrandsAgent
from strands.types.content import ContentBlock

from agent_tools.hooks import AgentLoggingHook
from agent_tools.image import capture_screenshot
from database.agents.specs import FIELD_SPECS, FieldSpec
from observability.instrumentation.gui_agent import GuiAction, GuiTaskTelemetry

MAX_ACTIONS = 40
"""How many actions one run may send to the robot."""

SETTLE_SECONDS = 1.0
"""The pause after an action, so the UI can settle before the next screenshot."""

GUI_ARGUMENTS: list[FieldSpec] = FIELD_SPECS.validate_python(
    [
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
)
"""The inputs of every GUI agent. A GUI agent's own ``arguments`` are replaced by these."""


class GuiRunReport(BaseModel):
    """What a GUI agent run did on the robot's desktop, built in code."""

    goal_reached: bool = Field(
        description="The agent called finished with success and stayed within the action cap."
    )
    summary: str = Field(description="The agent's finished summary, or why it stopped.")
    actions: list[GuiAction] = Field(description="Actions sent to the robot, in order.")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _check_box(box: list[float]) -> list[float]:
    if len(box) not in (2, 4):
        raise ValueError("give a point [x, y] or a box [x1, y1, x2, y2]")
    return box


type Box = Annotated[list[float], AfterValidator(_check_box)]

_BOX = "Point [x, y] or box [x1, y1, x2, y2] in the coordinate space given in the instructions."


type OriginalCoords = Callable[[float, float, int, int], list[float]]
"""Maps a model-space point to the screen, given the screen's width and height
(`Router.original_coords`)."""

type ToScreen = Callable[[Sequence[float]], list[float]]
"""Maps a model point, or a model box's center, to a screen pixel."""


@dataclass(frozen=True, slots=True)
class Screen:
    """The size of the last screenshot, which a model's coordinates are relative to."""

    width: int
    height: int

    def point(
        self, box: Sequence[float], original_coords: OriginalCoords
    ) -> list[float]:
        """The screen pixel of a model point, or of a model box's center."""
        if len(box) == 2:
            x, y = box
        else:
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return original_coords(x, y, self.width, self.height)


@dataclass(frozen=True, slots=True)
class RobotAction:
    """An action as the robot receives it in an ``action`` message."""

    action_type: str
    action_inputs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """What the model is shown after an action: the next observation, or an error."""

    content: list[ContentBlock]
    error: str | None = None

    @classmethod
    def failure(cls, error: str) -> ActionOutcome:
        return cls(content=[{"text": error}], error=error)


@dataclass(frozen=True, slots=True)
class GuiActionSpec:
    """One action a GUI agent can take.

    An action either goes to the robot (``to_robot``) or acts on the run itself
    (``run_locally``, for finishing and altering variables).
    """

    name: str
    description: str
    inputs: type[BaseModel]
    to_robot: Callable[[Any, ToScreen], RobotAction] | None = None
    run_locally: Callable[[GuiRun, Any], Awaitable[ActionOutcome]] | None = None

    def __post_init__(self) -> None:
        if (self.to_robot is None) == (self.run_locally is None):
            raise ValueError(
                f"The action '{self.name}' must either go to the robot or run locally."
            )

    def input_schema(self) -> dict[str, Any]:
        """The action's inputs as a JSON Schema."""
        schema = self.inputs.model_json_schema()
        _ = schema.pop("title", None)
        return schema


class GuiActions:
    """A set of actions, by name, in the order they are described to a model."""

    def __init__(self, *specs: GuiActionSpec) -> None:
        self._specs: dict[str, GuiActionSpec] = {spec.name: spec for spec in specs}

    def __iter__(self) -> Iterator[GuiActionSpec]:
        return iter(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def get(self, name: str) -> GuiActionSpec | None:
        return self._specs.get(name)


class PointInputs(BaseModel):
    start_box: Box = Field(description=_BOX)


class TextInputs(BaseModel):
    content: str = Field(
        description="Text to type. Use \\n at the end to submit (press Enter)."
    )


class HotkeyInputs(BaseModel):
    key: str = Field(
        description="Space-separated key names in lowercase, e.g. 'ctrl v' or 'ctrl shift s'."
    )


class KeyInputs(BaseModel):
    key: str = Field(description="Key name, e.g. 'shift', 'ctrl', 'alt'.")


class ScrollInputs(BaseModel):
    direction: Literal["up", "down", "left", "right"] = Field(
        description="The direction to scroll."
    )
    start_box: Box = Field(
        description=f"A point inside the area to scroll (the page, a list, or a specific element). {_BOX}"
    )


class DragInputs(BaseModel):
    start_box: Box = Field(description=f"Drag origin. {_BOX}")
    end_box: Box = Field(description=f"Drag destination. {_BOX}")


class WaitInputs(BaseModel):
    seconds: float = Field(description="Number of seconds to wait.")


class FinishedInputs(BaseModel):
    content: str = Field(description="Summary of what was accomplished.")
    success: bool = Field(
        default=True,
        description="Whether the recovery goal was reached; false when it cannot be.",
    )


class AlterVariableInputs(BaseModel):
    name: str = Field(description="The variable to alter.")
    value: Any = Field(description="Its new value, of the variable's JSON type.")


def _at_point(action_type: str) -> Callable[[PointInputs, ToScreen], RobotAction]:
    def to_robot(inputs: PointInputs, to_screen: ToScreen) -> RobotAction:
        return RobotAction(action_type, {"start_box": to_screen(inputs.start_box)})

    return to_robot


DEFAULT_GUI_ACTIONS = GuiActions(
    GuiActionSpec(
        "click",
        "Left-click on a UI element.",
        PointInputs,
        to_robot=_at_point("left_single"),
    ),
    GuiActionSpec(
        "double_click",
        "Double left-click on a UI element.",
        PointInputs,
        to_robot=_at_point("left_double"),
    ),
    GuiActionSpec(
        "right_click",
        "Right-click on a UI element.",
        PointInputs,
        to_robot=_at_point("right_single"),
    ),
    GuiActionSpec(
        "hover",
        "Move the mouse cursor over a UI element without clicking.",
        PointInputs,
        to_robot=_at_point("hover"),
    ),
    GuiActionSpec(
        "type_text",
        "Type text into the currently focused input field.",
        TextInputs,
        to_robot=lambda inputs, _to_screen: RobotAction(
            "type", {"content": inputs.content}
        ),
    ),
    GuiActionSpec(
        "hotkey",
        "Press a keyboard shortcut (e.g. 'ctrl c', 'alt f4', 'ctrl -' to zoom out, 'ctrl +' to zoom in). "
        "Split keys with a space, use lowercase, max 3 keys. Key names: ctrl, shift, alt, win, enter, "
        "esc, tab, space, backspace, delete, up, down, left, right, pageup, pagedown, home, end, f1-f12, "
        "or the character itself (a, 1, -, +, =).",
        HotkeyInputs,
        to_robot=lambda inputs, _to_screen: RobotAction(
            "hotkey", {"hotkey": inputs.key}
        ),
    ),
    GuiActionSpec(
        "key_down",
        "Press and hold a keyboard key.",
        KeyInputs,
        to_robot=lambda inputs, _to_screen: RobotAction("keydown", {"key": inputs.key}),
    ),
    GuiActionSpec(
        "key_up",
        "Release a keyboard key that was held down.",
        KeyInputs,
        to_robot=lambda inputs, _to_screen: RobotAction("keyup", {"key": inputs.key}),
    ),
    GuiActionSpec(
        "scroll",
        "Scroll in the given direction ('up', 'down', 'left', 'right') at a screen location.",
        ScrollInputs,
        to_robot=lambda inputs, to_screen: RobotAction(
            "scroll",
            {
                "direction": inputs.direction,
                "start_box": to_screen(inputs.start_box),
            },
        ),
    ),
    GuiActionSpec(
        "drag",
        "Click and drag from one screen location to another.",
        DragInputs,
        to_robot=lambda inputs, to_screen: RobotAction(
            "drag",
            {
                "start_box": to_screen(inputs.start_box),
                "end_box": to_screen(inputs.end_box),
            },
        ),
    ),
    GuiActionSpec(
        "wait",
        "Wait for a specified number of seconds.",
        WaitInputs,
        to_robot=lambda inputs, _to_screen: RobotAction(
            "wait", {"seconds": inputs.seconds}
        ),
    ),
    GuiActionSpec(
        "finished",
        "Signal that the recovery task is complete. Call it when the recovery goal is "
        "reached, or when it clearly cannot be reached.",
        FinishedInputs,
        run_locally=lambda run, inputs: run.finish(inputs),
    ),
    GuiActionSpec(
        "alter_variable",
        "Alter the value of a given variable. The value must be JSON serializable "
        "and keep the variable's JSON type.",
        AlterVariableInputs,
        run_locally=lambda run, inputs: run.alter_variable(inputs),
    ),
)
"""The actions every GUI agent has, whichever loop runs it."""


def coerce_to_held_type(held: Any, value: Any) -> Any:
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


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GuiRun:
    """One GUI run: the robot and screen it acts on, and what it ended with."""

    actions: GuiActions
    telemetry: GuiTaskTelemetry
    websocket: Any
    hook: AgentLoggingHook
    original_coords: OriginalCoords
    observe: Callable[[bytes], list[ContentBlock]]
    """What the model is shown of a screenshot taken after an action."""
    variables: dict[str, Any]
    image: bytes
    screen: Screen
    agent: StrandsAgent | None = None
    sent: int = 0
    cap_reached: bool = False
    summary: str | None = None
    success: bool = False
    disconnect: WebSocketDisconnect | None = None
    error: str | None = None

    @property
    def over(self) -> bool:
        """Whether the run finished, reached the action cap or lost the robot."""
        return (
            self.summary is not None or self.cap_reached or self.disconnect is not None
        )

    def to_screen(self, box: Sequence[float]) -> list[float]:
        return self.screen.point(box, self.original_coords)

    async def perform(self, name: str, raw_inputs: Mapping[str, Any]) -> ActionOutcome:
        """Take the action called ``name`` and return what the model sees next.

        An unknown action, invalid inputs, the action cap and a failure on the robot
        come back as an error outcome, so the model can react.

        Raises:
            WebSocketDisconnect: when the robot disconnects; the run records it.
        """
        spec = self.actions.get(name)
        if spec is None:
            return ActionOutcome.failure(
                f"Unknown action '{name}'. Use one of: {', '.join(self.actions.names())}."
            )
        try:
            inputs = spec.inputs.model_validate(dict(raw_inputs))
        except ValidationError as exc:
            return ActionOutcome.failure(f"Invalid inputs for '{name}': {exc}")
        if spec.to_robot is not None:
            return await self._send(spec.to_robot(inputs, self.to_screen))
        if spec.run_locally is not None:
            return await spec.run_locally(self, inputs)
        raise TypeError(
            f"The action '{name}' neither goes to the robot nor runs locally."
        )

    async def _send(self, action: RobotAction) -> ActionOutcome:
        if self.sent >= MAX_ACTIONS:
            if not self.cap_reached:
                self.cap_reached = True
                self.telemetry.max_actions_exceeded(self.telemetry.actions, MAX_ACTIONS)
            return ActionOutcome.failure(
                f"Action limit of {MAX_ACTIONS} reached: no more actions are sent and the run ends."
            )
        self.sent += 1
        try:
            return await self._dispatch(action)
        except WebSocketDisconnect as exc:
            self.disconnect = exc
            raise

    async def _dispatch(self, action: RobotAction) -> ActionOutcome:
        """Send one action, record its GUI trace, and return a fresh screenshot."""
        telemetry = self.telemetry
        action_type, action_inputs = action.action_type, action.action_inputs
        with telemetry.action(telemetry.actions + 1, action_type) as action_telemetry:
            started_at = datetime.now(UTC)
            await self.websocket.send_json(
                {
                    "type": "action",
                    "content": {
                        "action_type": action_type,
                        "action_inputs": action_inputs,
                    },
                }
            )
            result = await self.websocket.receive_json()
            finished_at = datetime.now(UTC)

            success = bool(result.get("success", False))
            action_telemetry.result(success)
            await self.hook.register_gui_trace(
                action_type,
                action_inputs,
                self.image,
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
                return ActionOutcome.failure(
                    f"Action '{action_type}' failed on client side. Inputs: {action_inputs}. "
                    f"Client error: {result.get('error') or 'unknown'}"
                )

        await asyncio.sleep(SETTLE_SECONDS)
        self.image, (width, height) = await capture_screenshot(
            self.websocket, telemetry
        )
        self.screen = Screen(width, height)
        return ActionOutcome(content=self.observe(self.image))

    async def finish(self, inputs: FinishedInputs) -> ActionOutcome:
        """End the run with the model's summary and record the final GUI trace."""
        now = datetime.now(UTC)
        self.summary = inputs.content
        self.success = inputs.success
        if not self.cap_reached:
            self.telemetry.done(self.telemetry.actions)
        await self.hook.register_gui_trace(
            "finished", {"content": inputs.content}, self.image, now, now, success=True
        )
        return ActionOutcome(content=[{"text": f"Task completed: {inputs.content}"}])

    async def alter_variable(self, inputs: AlterVariableInputs) -> ActionOutcome:
        """Set a Variable the robot continues with, keeping its JSON type."""
        if inputs.name not in self.variables:
            return ActionOutcome.failure(f"Variable '{inputs.name}' not found.")
        try:
            value = coerce_to_held_type(self.variables[inputs.name], inputs.value)
        except ValueError as exc:
            return ActionOutcome.failure(f"Variable '{inputs.name}': {exc}")
        self.variables[inputs.name] = value
        return ActionOutcome(
            content=[
                {
                    "text": f"Variable '{inputs.name}' has been updated to {json.dumps(value)}."
                }
            ]
        )

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
        )

    def transcript(self) -> list[list[ContentBlock]]:
        """The content of the model's messages, in order, after the summary of older ones if any."""
        if self.agent is None:
            return []
        summary = self.agent.conversation_manager.get_state().get("summary_message")
        return [
            m["content"]
            for m in self.agent.messages
            if m["role"] == "assistant" or m is summary
        ]
