"""A GUI agent is a parent agent's tool, named and described by its own row."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest
from strands import Agent as StrandsAgent

from database.agents.behaviours.gui.shared import GUI_ARGUMENTS
from tests.unit.shared.gui_fakes import (
    GUI_TASK_INPUT,
    make_gui_agent,
    make_robot,
    patch_gui_io,
)
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall

pytestmark = pytest.mark.asyncio


def _tool_results(model: ScriptedModel) -> dict[str, dict[str, Any]]:
    """The parent's tool results, by the id of the call they answer."""
    return {
        block["toolResult"]["toolUseId"]: block["toolResult"]
        for message in model.requests[-1]
        for block in message["content"]
        if "toolResult" in block
    }


async def test_two_gui_sub_agents_are_two_tools_that_report_to_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_gui_io(monkeypatch)
    clicker = make_gui_agent(
        ToolCall("alter_variable", {"name": "total", "value": "42"}),
        ToolCall("finished", {"content": "Stored the total"}),
        name="Clicker",
        description="Clicks through the invoice form.",
    )
    filler = make_gui_agent(
        ConnectionError("model endpoint unreachable"),
        name="Form Filler",
        description="Fills the invoice form.",
    )
    inputs = GUI_TASK_INPUT | {"variables": {"total": "0"}}
    parent_model = ScriptedModel(
        ToolCall("clicker", inputs), ToolCall("form_filler", inputs), "Recovered."
    )
    parent = StrandsAgent(
        model=parent_model,
        tools=[clicker.as_tool(), filler.as_tool()],
        callback_handler=None,
    )

    _ = await parent.invoke_async(
        "Recover the robot",
        invocation_state={"websocket": make_robot(), "robot_exception_id": uuid4()},
    )

    specs = {spec["name"]: spec for spec in parent_model.tool_specs[0]}
    assert specs["clicker"]["description"] == "Clicks through the invoice form."
    assert specs["form_filler"]["description"] == "Fills the invoice form."
    for spec in specs.values():
        properties = spec["inputSchema"]["json"]["properties"]
        assert list(properties) == [field.name for field in GUI_ARGUMENTS]

    results = _tool_results(parent_model)
    [block] = results["call-1"]["content"]
    assert json.loads(block["text"]) == {
        "goal_reached": True,
        "summary": "Stored the total",
        "actions": [],
        "messages": [
            [
                {
                    "toolUse": {
                        "toolUseId": "call-1",
                        "name": "alter_variable",
                        "input": {"name": "total", "value": "42"},
                    }
                }
            ],
            [
                {
                    "toolUse": {
                        "toolUseId": "call-2",
                        "name": "finished",
                        "input": {"content": "Stored the total"},
                    }
                }
            ],
        ],
        "variables": {"total": "42"},
    }
    [error] = results["call-2"]["content"]
    assert error == {"text": "model endpoint unreachable"}
