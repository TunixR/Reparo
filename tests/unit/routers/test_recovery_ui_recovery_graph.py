"""A recovery runs the seeded "UI Recovery" graph end to end over the WebSocket.

The agents and the graph come from the real populators. Each agent runs a real
Strands agent over a scripted model, and the GUI agent drives the test's robot
through screenshot and action frames.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from database.agent_graphs.runner import NodeRun
from database.agents.models import Agent
from database.populators.agent_graphs import populate_agent_graphs
from database.provider.models import Router
from tests.unit.agent_tools._shared.process_models import SPENDING_APPROVAL_MODEL
from tests.unit.shared.fake_websocket import make_test_image_bytes
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn
from tests.unit.shared.tooled_gui_fakes import patch_gui_io

GATEWAY = "Gateway Orchestrator"
INFERER = "Recovery Inferer"
GUI = "Tooled GuiAgent"
CONTINUATION = "Continuation Inferer"


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-ui-recovery-01"
    _ = make_robot_key(name="UI Recovery", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def payload(valid_recovery_payload: dict[str, Any]) -> dict[str, Any]:
    return valid_recovery_payload | {"model": SPENDING_APPROVAL_MODEL}


@pytest.fixture
def ui_recovery(engine: Engine, seeded_agents: dict[str, Agent]) -> None:
    populate_agent_graphs(engine)


@pytest.fixture
def script(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., dict[str, ScriptedModel]]:
    """Give each seeded agent, by name, a scripted model."""

    def _script(turns: Mapping[str, list[Turn]]) -> dict[str, ScriptedModel]:
        models = {
            name: ScriptedModel(*turns.get(name, ["Unused."]))
            for name in (GATEWAY, INFERER, GUI, CONTINUATION)
        }
        monkeypatch.setattr(Agent, "get_model", lambda self: models[self.name])  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        monkeypatch.setattr(Agent, "get_logging_hooks", lambda self, state: [])  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(Router, "get_conversation_cost", lambda self, i, o: 0.0)  # pyright: ignore[reportUnknownLambdaType]
        patch_gui_io(monkeypatch)
        return models

    return _script


@pytest.fixture
def node_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The agent of each node run, in order."""
    runs: list[str] = []
    original = Agent.run_node

    async def _spy(
        self: Agent, invocation_state: dict[str, Any], input_map: Mapping[str, str]
    ) -> NodeRun:
        runs.append(self.name)
        return await original(self, invocation_state, input_map)

    monkeypatch.setattr(Agent, "run_node", _spy)
    return runs


def _recover(
    client: TestClient, robot_key_raw: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Play the robot: the answer, and the actions it was asked to perform."""
    actions: list[dict[str, Any]] = []
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(payload)
        while True:
            message = ws.receive_json()
            match message["type"]:
                case "ping":
                    continue
                case "screenshot":
                    ws.send_bytes(make_test_image_bytes(size=(64, 48)))
                case "action":
                    actions.append(message)
                    ws.send_json({"success": True})
                case _:
                    return message, actions


def _texts(model: ScriptedModel) -> str:
    """Every user text a model was sent."""
    return "\n".join(
        block["text"]
        for request in model.requests
        for message in request
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    )


def _classified(*, ui_exception: bool) -> list[Turn]:
    return [
        "Classified.",
        ToolCall(
            "GatewayDecision",
            {"ui_exception": ui_exception, "rationale": "The Approve click failed."},
        ),
    ]


INFERRED: list[Turn] = [
    "Inferred.",
    ToolCall(
        "RecoveryInference",
        {
            "failure_kind": "navigation",
            "failed_model_event_id": "click_approve",
            "loop_passes": [2],
            "recovery_task": "Approve the current spending request",
            "rationale": "A later pass, so only the current request.",
        },
    ),
]
RECOVERED_ON_THE_DESKTOP: list[Turn] = [
    ToolCall("click", {"start_box": [500, 500]}),
    ToolCall("finished", {"content": "Request approved"}),
]
CONTINUED: list[Turn] = [
    ToolCall(
        "set_continuation",
        {
            "recovered": True,
            "process_finished": False,
            "model_event_id": "back_to_list",
            "loop_iteration": [2],
        },
    ),
    "Continuation set.",
    ToolCall(
        "ContinuationReport",
        {"success": True, "summary": "Approved; the robot returns to the list."},
    ),
]


@pytest.mark.usefixtures("ui_recovery")
def test_a_ui_failure_runs_all_four_nodes_and_is_recovered(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[str],
):
    models = script(
        {
            GATEWAY: _classified(ui_exception=True),
            INFERER: INFERRED,
            GUI: RECOVERED_ON_THE_DESKTOP,
            CONTINUATION: CONTINUED,
        }
    )

    message, actions = _recover(client, robot_key_raw, payload)

    assert node_runs == [GATEWAY, INFERER, GUI, CONTINUATION]
    assert message["type"] == "done"
    assert message["content"] == {
        "success": True,
        "process_finished": False,
        "continue_from_model_event_id": "back_to_list",
        "loop_iteration": [2],
        "variables": payload["variables"],
    }
    assert len(actions) == 1
    assert "Task: Approve the current spending request" in _texts(models[GUI])
    continuation_inputs = _texts(models[CONTINUATION])
    assert "recovery_task: Approve the current spending request" in continuation_inputs
    assert "gui_summary: Request approved" in continuation_inputs
    assert "goal_reached: True" in continuation_inputs


@pytest.mark.usefixtures("ui_recovery")
def test_a_failure_that_is_no_ui_exception_runs_only_the_gateway(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[str],
):
    models = script({GATEWAY: _classified(ui_exception=False)})

    message, actions = _recover(client, robot_key_raw, payload)

    assert node_runs == [GATEWAY]
    assert message["type"] == "done"
    assert message["content"]["success"] is False
    assert actions == []
    assert all(models[name].calls == 0 for name in (INFERER, GUI, CONTINUATION))
