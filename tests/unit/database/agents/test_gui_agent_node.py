"""A GUI agent row runs as a graph node and reports what happened on the desktop."""

from collections.abc import Callable
from typing import Any, override

import pytest
from fastapi import WebSocketDisconnect
from sqlmodel import Session

import database.agents.tooled_gui_agent as tooled_module
from database.agent_graphs.runner import NodeRun
from database.agents.models import Agent, AgentType, Argument
from database.agents.tooled_gui_agent import GuiAction, GuiRunReport
from database.provider.models import Router
from tests.unit.fixtures.agent_fixtures import node_state
from tests.unit.shared.fake_websocket import FakeWebSocket, make_test_image_bytes
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn
from tests.unit.shared.tooled_gui_fakes import GUI_TASK_INPUT, make_robot, patch_gui_io

pytestmark = pytest.mark.asyncio

type GuiNode = Callable[..., Agent]
"""Factory: model turns in, a persisted GUI agent row whose model plays them out."""

GUI_ARGUMENTS = {
    "task": ("str", "string"),
    "variables": ("dict", "object"),
    "ui_log": ("list", "array"),
    "errored_act": ("dict", "object"),
    "model": ("str", "string"),
}


@pytest.fixture
def gui_node(
    session: Session,
    make_agent: Callable[..., Agent],
    mock_router: Router,
    monkeypatch: pytest.MonkeyPatch,
) -> GuiNode:
    patch_gui_io(monkeypatch)
    agent = make_agent(
        router=mock_router,
        type=AgentType.GuiAgent,
        prompt="{instruction}",
        arguments=[
            Argument(name=name, description=name, type=py, json_type=js)
            for name, (py, js) in GUI_ARGUMENTS.items()
        ],
    )

    def _gui_node(*turns: Turn) -> Agent:
        model = ScriptedModel(*turns)
        monkeypatch.setattr(Router, "get_model", lambda _self, **_params: model)
        return agent

    return _gui_node


def _inputs(**overrides: Any) -> dict[str, Any]:
    return GUI_TASK_INPUT | overrides


async def test_report_lists_the_actions_and_the_finished_summary(
    gui_node: GuiNode,
) -> None:
    agent = gui_node(
        ToolCall("click", {"start_box": [10, 20]}),
        ToolCall("type_text", {"content": "42"}),
        ToolCall("finished", {"content": "Invoice filled"}),
    )
    robot = make_robot()

    run = await agent.run_node(node_state(_inputs(), websocket=robot), {})

    assert isinstance(run, NodeRun)
    assert run.output == GuiRunReport(
        goal_reached=True,
        summary="Invoice filled",
        actions=[
            GuiAction(action_type="left_single", success=True),
            GuiAction(action_type="type", success=True),
        ],
    )
    assert len(robot.actions) == 2


async def test_output_model_of_a_gui_agent_is_the_run_report(
    gui_node: GuiNode,
) -> None:
    assert gui_node("unused").output_model() is GuiRunReport


async def test_action_cap_ends_the_run_without_reaching_the_goal(
    gui_node: GuiNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tooled_module, "MAX_ACTIONS", 2)
    # The model keeps clicking; a success claim after the cap doesn't count.
    agent = gui_node(
        ToolCall("click", {"start_box": [1, 1]}),
        ToolCall("click", {"start_box": [2, 2]}),
        ToolCall("click", {"start_box": [3, 3]}),
        ToolCall("finished", {"content": "Done", "success": True}),
    )
    robot = make_robot()

    run = await agent.run_node(node_state(_inputs(), websocket=robot), {})

    assert isinstance(run.output, GuiRunReport)
    assert run.output.goal_reached is False
    assert (
        run.output.summary
        == "Stopped at the cap of 2 actions without calling finished."
    )
    assert [a.action_type for a in run.output.actions] == ["left_single"] * 2
    assert len(robot.actions) == 2


async def test_finished_without_success_reports_the_goal_not_reached(
    gui_node: GuiNode,
) -> None:
    agent = gui_node(
        ToolCall("finished", {"content": "The form is gone", "success": False})
    )

    run = await agent.run_node(node_state(_inputs(), websocket=make_robot()), {})

    assert run.output == GuiRunReport(
        goal_reached=False, summary="The form is gone", actions=[]
    )


class DisconnectingRobot(FakeWebSocket):
    """Serves screenshots, then disconnects instead of answering an action."""

    @override
    async def receive_json(self) -> dict[str, Any]:
        raise WebSocketDisconnect(code=1006)


async def test_robot_disconnect_propagates(gui_node: GuiNode) -> None:
    agent = gui_node(
        ToolCall("click", {"start_box": [10, 20]}),
        ToolCall("finished", {"content": "unreachable"}),
    )
    robot = DisconnectingRobot([make_test_image_bytes(size=(64, 48))])

    with pytest.raises(WebSocketDisconnect):
        _ = await agent.run_node(node_state(_inputs(), websocket=robot), {})


async def test_model_error_propagates(gui_node: GuiNode) -> None:
    agent = gui_node(ConnectionError("model endpoint unreachable"))

    with pytest.raises(ConnectionError, match="model endpoint unreachable"):
        _ = await agent.run_node(node_state(_inputs(), websocket=make_robot()), {})
