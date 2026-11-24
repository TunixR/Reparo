"""Every GUI agent has the same actions, each sent to the robot as its own message."""

from __future__ import annotations

from typing import Any

import pytest

import database.agents.behaviours.gui.shared as gui_shared
from database.agents.behaviours import TextGuiAgent, TooledGuiAgent
from database.agents.behaviours.gui import GuiAgent
from database.agents.behaviours.gui.shared import DEFAULT_GUI_ACTIONS
from tests.unit.shared.gui_fakes import (
    GUI_KINDS,
    clicking_gui_agent,
    drive,
    fake_router,
    make_gui_agent,
    make_robot,
    patch_gui_io,
    scripted_model,
)
from tests.unit.shared.scripted_model import ToolCall

ACTION_NAMES = [
    "click",
    "double_click",
    "right_click",
    "hover",
    "type_text",
    "hotkey",
    "key_down",
    "key_up",
    "scroll",
    "drag",
    "wait",
    "finished",
    "alter_variable",
]


def test_both_gui_agents_take_the_default_actions() -> None:
    assert TooledGuiAgent.actions.names() == ACTION_NAMES
    assert TextGuiAgent.actions.names() == ACTION_NAMES


@pytest.mark.asyncio
async def test_the_tooled_agent_has_a_tool_per_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_gui_io(monkeypatch)
    gui_agent = make_gui_agent(ToolCall("finished", {"content": "done"}))

    _ = await drive(gui_agent, make_robot())

    [tool_specs] = scripted_model(gui_agent).tool_specs
    assert [spec["name"] for spec in tool_specs] == ACTION_NAMES
    descriptions = {spec.name: spec.description for spec in DEFAULT_GUI_ACTIONS}
    assert {s["name"]: s["description"] for s in tool_specs} == descriptions


# The model answers in a 1000x1000 space; screenshots are 64x48.
MODEL_SPACE = fake_router(
    relative_dim=lambda _h, _w: [1000, 1000],
    original_coords=lambda x, y, w, h: [x * w / 1000, y * h / 1000],
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "inputs", "action_type", "action_inputs"),
    [
        ("click", {"start_box": [500, 500]}, "left_single", {"start_box": [32, 24]}),
        (
            "double_click",
            {"start_box": [0, 1000]},
            "left_double",
            {"start_box": [0, 48]},
        ),
        (
            "right_click",
            {"start_box": [250, 500, 750, 1000]},
            "right_single",
            {"start_box": [32, 36]},
        ),
        ("hover", {"start_box": [500, 0]}, "hover", {"start_box": [32, 0]}),
        ("type_text", {"content": "42\n"}, "type", {"content": "42\n"}),
        ("hotkey", {"key": "ctrl c"}, "hotkey", {"hotkey": "ctrl c"}),
        ("key_down", {"key": "shift"}, "keydown", {"key": "shift"}),
        ("key_up", {"key": "shift"}, "keyup", {"key": "shift"}),
        (
            "scroll",
            {"direction": "down", "start_box": [500, 500]},
            "scroll",
            {"direction": "down", "start_box": [32, 24]},
        ),
        (
            "drag",
            {"start_box": [0, 0], "end_box": [1000, 1000]},
            "drag",
            {"start_box": [0, 0], "end_box": [64, 48]},
        ),
        ("wait", {"seconds": 2}, "wait", {"seconds": 2.0}),
    ],
)
async def test_each_action_is_one_robot_message(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    inputs: dict[str, Any],
    action_type: str,
    action_inputs: dict[str, Any],
) -> None:
    patch_gui_io(monkeypatch)
    gui_agent = make_gui_agent(
        ToolCall(name, inputs),
        ToolCall("finished", {"content": "done"}),
        router=MODEL_SPACE,
    )
    robot = make_robot()

    run = await drive(gui_agent, robot)

    assert robot.actions == [
        {
            "type": "action",
            "content": {"action_type": action_type, "action_inputs": action_inputs},
        }
    ]
    assert run.report().goal_reached


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "inputs"),
    [
        ("click", {}),
        ("click", {"start_box": [1, 2, 3]}),
        ("scroll", {"direction": "sideways", "start_box": [1, 2]}),
        ("teleport", {}),
    ],
)
async def test_an_invalid_action_is_answered_with_an_error_and_not_sent(
    monkeypatch: pytest.MonkeyPatch, name: str, inputs: dict[str, Any]
) -> None:
    patch_gui_io(monkeypatch)
    gui_agent = make_gui_agent(
        ToolCall(name, inputs), ToolCall("finished", {"content": "done"})
    )
    robot = make_robot()

    _ = await drive(gui_agent, robot)

    assert robot.actions == []
    tool_result = scripted_model(gui_agent).requests[1][-1]["content"][0]["toolResult"]
    assert tool_result["status"] == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", GUI_KINDS)
async def test_both_loops_stop_at_the_action_cap(
    monkeypatch: pytest.MonkeyPatch, kind: type[GuiAgent]
) -> None:
    patch_gui_io(monkeypatch)
    monkeypatch.setattr(gui_shared, "MAX_ACTIONS", 2)
    robot = make_robot()

    run = await drive(clicking_gui_agent(kind), robot)

    assert len(robot.actions) == 2
    assert run.telemetry.outcome == "max_actions_exceeded"
    report = run.report()
    assert not report.goal_reached
    assert report.summary == "Stopped at the cap of 2 actions without calling finished."
