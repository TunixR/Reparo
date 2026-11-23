"""Doubles for driving a GUI agent without a robot, Postgres or an LLM."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, override
from uuid import uuid4

import pytest
from strands.hooks import HookProvider, HookRegistry

import database.agents.behaviours.base as behaviour_base
import database.agents.behaviours.gui.shared as gui_shared
from database.agents.behaviours.gui import GuiAgent, TooledGuiAgent
from database.agents.behaviours.gui.shared import GUI_ARGUMENTS, GuiRun
from database.agents.models import Agent
from tests.unit.shared.fake_websocket import FakeWebSocket, make_test_image_bytes
from tests.unit.shared.scripted_model import ScriptedModel, Turn


class FakeHook(HookProvider):
    """Stands in for AgentLoggingHook, which writes audit rows to Postgres."""

    def __init__(self, **_: Any) -> None:
        self.gui_traces: list[tuple[str, bool]] = []

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        return None

    async def register_gui_trace(
        self,
        action_type: str,
        _content: Any,
        _screenshot: bytes,
        _started_at: Any,
        _finished_at: Any,
        success: bool = True,
    ) -> None:
        self.gui_traces.append((action_type, success))

    def update_trace(self, finished: bool = False, cost: float = 0.0) -> None:
        return None


UITARS_PARSER = "database.agents.behaviours.gui.parsers.uitars:UITarsParser"


def make_robot(*, action_success: bool = True) -> FakeWebSocket:
    """A robot that serves 64x48 screenshots and answers actions with ``action_success``."""
    screenshots = [make_test_image_bytes(size=(64, 48)) for _ in range(10)]
    return FakeWebSocket(screenshots, action_success=action_success)


def patch_gui_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the settle delay between actions and the audit rows."""

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(behaviour_base, "AgentLoggingHook", FakeHook)
    monkeypatch.setattr(gui_shared, "asyncio", SimpleNamespace(sleep=_no_sleep))


def fake_router(
    *,
    relative_dim: Callable[[int, int], list[int]] = lambda h, w: [h, w],
    original_coords: Callable[[float, float, int, int], list[float]] = (
        lambda x, y, _w, _h: [x, y]
    ),
    model_name: str = "scripted",
) -> SimpleNamespace:
    """A router whose model space is ``relative_dim`` of the screenshot, mapped back
    through ``original_coords``."""
    return SimpleNamespace(
        relative_dim=relative_dim,
        original_coords=original_coords,
        model_name=model_name,
        get_conversation_cost=lambda _i, _o: 0.0,
    )


def make_gui_agent[T: GuiAgent](
    *turns: Turn,
    kind: type[T] = TooledGuiAgent,
    router: SimpleNamespace | None = None,
    prompt: str = "{instruction}",
    parser_path: str | None = None,
    name: str = "Test GUI Agent",
    description: str = "A GUI agent for tests.",
) -> T:
    """A GUI agent of ``kind`` whose inner Strands model plays ``turns``.

    Its row is a stand-in: a real `Agent` validates its inputs, and the stand-in
    gives it ``router`` and the scripted model.
    """
    model = ScriptedModel(*turns)
    row = Agent(
        name=name,
        description=description,
        prompt=prompt,
        type=kind.agent_type,
        input_type=Agent.InputType.IMAGETEXT,
        arguments=list(GUI_ARGUMENTS),
        parser_path=parser_path,
    )
    agent = SimpleNamespace(
        id=uuid4(),
        name=name,
        description=row.description,
        prompt=prompt,
        parser_path=parser_path,
        arguments=row.arguments,
        validate_input=row.validate_input,
        get_input_schema=row.get_input_schema,
        router=router or fake_router(),
        get_model=lambda: model,
    )
    return kind(agent)  # pyright: ignore[reportArgumentType]


def scripted_model(gui_agent: GuiAgent) -> ScriptedModel:
    """The scripted model ``gui_agent`` was made with."""
    return gui_agent.agent.get_model()  # pyright: ignore[reportReturnType]


GUI_TASK_INPUT: dict[str, Any] = {
    "task": "Fill the invoice",
    "variables": {},
    "ui_log": [{}, {}],
    "errored_act": {"activity_id": 3},
    "model": "invoice-process",
}


async def drive(
    gui_agent: GuiAgent,
    robot: FakeWebSocket,
    *,
    state: dict[str, Any] | None = None,
    **inputs: Any,
) -> GuiRun:
    """Run ``gui_agent`` standalone on `GUI_TASK_INPUT` with ``inputs`` over it."""
    return await gui_agent.drive(
        {"websocket": robot, **(state or {})},
        GUI_TASK_INPUT | inputs,
        contain_errors=True,
    )
