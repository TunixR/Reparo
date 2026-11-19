from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from agent_tools.utils import CONTINUATION_STATE_KEY, Continuation
from database.agent_graphs.runner import AGENT_GRAPH_CONTEXT_KEY, NodeRun
from database.agents.models import Agent
from database.logging.models import RecoveryContext, RobotException
from robot.capabilities import CAPABILITIES_STATE_KEY
from templates.gateway import ResponseToRPA
from templates.recovery_payload import RecoveryInputs, RecoveryPayload


@pytest.fixture
def valid_robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-payload-0001"
    _ = make_robot_key(name="Enabled", enabled=True, key_raw=key_raw)
    return key_raw


def test_recovery_ws_ingest_stores_raw_payload_and_normalized_context(
    session: Session,
    client: TestClient,
    valid_robot_key_raw: str,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    monkeypatch,  # pyright: ignore[reportMissingParameterType]
):
    captured_invocation: dict[str, Any] = {}
    captured_from_payload: dict[str, Any] = {}

    original_from_payload = RecoveryContext.from_payload

    def _spy_from_payload(cls, *, payload):  # pyright: ignore[reportMissingParameterType,reportUnusedParameter]
        captured_from_payload["payload"] = payload
        return original_from_payload(payload=payload)

    monkeypatch.setattr(
        RecoveryContext,
        "from_payload",
        classmethod(_spy_from_payload),
        raising=True,
    )

    async def _fake_run_node(_self, invocation_state, _input_map):  # pyright: ignore[reportMissingParameterType]
        captured_invocation["invocation_state"] = invocation_state
        captured_invocation["inputs"] = invocation_state[
            AGENT_GRAPH_CONTEXT_KEY
        ].recovery_inputs
        invocation_state[CONTINUATION_STATE_KEY] = Continuation(
            process_finished=False,
            continue_from_model_event_id="click_submit",
            loop_iteration=[],
        )
        return NodeRun(output=ResponseToRPA(success=True))

    monkeypatch.setattr(Agent, "run_node", _fake_run_node, raising=True)

    headers = {"X-ROBOT-KEY": valid_robot_key_raw}
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers=headers
    ) as ws:
        ws.send_json(valid_recovery_payload)
        message = ws.receive_json()

    assert message["type"] == "done"
    assert isinstance(message["content"], dict)
    assert message["content"] == {
        "success": True,
        "process_finished": False,
        "continue_from_model_event_id": "click_submit",
        "loop_iteration": [],
        "variables": valid_recovery_payload["variables"],
    }

    robot_exception = session.get(RobotException, UUID(message["id"]))
    assert robot_exception is not None
    assert robot_exception.exception_details == valid_recovery_payload

    recovery_context = session.exec(
        select(RecoveryContext).where(
            RecoveryContext.robot_exception_id == robot_exception.id
        )
    ).first()
    assert recovery_context is not None
    assert recovery_context.task_name == valid_recovery_payload["task_name"]
    assert recovery_context.platform == valid_recovery_payload["platform"]
    assert recovery_context.os == valid_recovery_payload["os"]
    assert recovery_context.variables == valid_recovery_payload["variables"]
    assert recovery_context.model == valid_recovery_payload["model"]
    assert len(recovery_context.ui_log_entries) == 1
    assert recovery_context.ui_log_entries[0].activity_id is None
    assert recovery_context.ui_log_entries[0].event_name == "Type invoice id"
    assert recovery_context.ui_log_entries[0].model_event_id == "type_invoice_id"
    assert recovery_context.errored_activity is not None
    assert recovery_context.errored_activity.activity_id is None
    assert recovery_context.errored_activity.event_name is None
    assert recovery_context.errored_activity.error_code == "E500"
    assert recovery_context.errored_activity.model_event_id == "click_submit"

    assert isinstance(captured_from_payload["payload"], RecoveryPayload)
    # The fixture declares no capabilities, so the vision-only default applies,
    # and no recovery task.
    parsed_payload = captured_from_payload["payload"]
    assert parsed_payload.recovery_task is None
    assert (
        parsed_payload.model_dump(exclude={"capabilities", "recovery_task"})
        == valid_recovery_payload
    )
    default_capabilities = {"modalities": ["vision"], "ocr": False, "details": {}}
    assert recovery_context.capabilities == default_capabilities
    assert (
        captured_invocation["invocation_state"][CAPABILITIES_STATE_KEY]
        == default_capabilities
    )

    assert (
        captured_invocation["invocation_state"]["robot_exception_id"]
        == robot_exception.id
    )
    state_variables = captured_invocation["invocation_state"]["variables"]
    assert state_variables == valid_recovery_payload["variables"]
    assert state_variables is not recovery_context.variables
    inputs = captured_invocation["inputs"]
    assert RecoveryInputs.model_validate(inputs).model_dump() == inputs
    assert (
        captured_invocation["inputs"]["task_name"]
        == valid_recovery_payload["task_name"]
    )
    assert (
        captured_invocation["inputs"]["platform"] == valid_recovery_payload["platform"]
    )
    assert captured_invocation["inputs"]["os"] == valid_recovery_payload["os"]
    assert (
        captured_invocation["inputs"]["variables"]
        == valid_recovery_payload["variables"]
    )
    # Check that ui_log uses v2 field names with normalized ids
    ui_log = captured_invocation["inputs"]["ui_log"]
    assert len(ui_log) == 1
    assert ui_log[0]["activity_id"] is None  # fixture has activity_id as None
    assert ui_log[0]["event_name"] == "Type invoice id"
    assert ui_log[0]["case_id"] == valid_recovery_payload["ui_log"][0]["case_id"]
    assert ui_log[0]["event_id"] == valid_recovery_payload["ui_log"][0]["event_id"]
    assert ui_log[0]["model_event_id"] == "type_invoice_id"
    # Check that all id fields are strings
    assert isinstance(ui_log[0]["case_id"], str)
    assert isinstance(ui_log[0]["event_id"], str)

    # Check that errored_act uses v2 field names with normalized ids
    errored_act = captured_invocation["inputs"]["errored_act"]
    assert errored_act["activity_id"] is None  # fixture has activity_id as None
    assert errored_act["event_name"] is None  # fixture has event_name as None
    assert errored_act["case_id"] == valid_recovery_payload["errored_act"]["case_id"]
    assert errored_act["event_id"] == valid_recovery_payload["errored_act"]["event_id"]
    assert errored_act["model_event_id"] == "click_submit"
    assert errored_act["activity_name"] == "Click Submit"
    # Check that all id fields are strings
    assert isinstance(errored_act["case_id"], str)
    assert isinstance(errored_act["event_id"], str)

    assert captured_invocation["inputs"]["model"] == valid_recovery_payload["model"]
    assert set(captured_invocation["inputs"].keys()) == {
        "task_name",
        "platform",
        "os",
        "variables",
        "ui_log",
        "errored_act",
        "model",
    }


def test_recovery_ws_ingest_stores_declared_capabilities(
    session: Session,
    client: TestClient,
    valid_robot_key_raw: str,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    monkeypatch,  # pyright: ignore[reportMissingParameterType]
):
    capabilities = {
        "modalities": ["dom", "vision"],
        "ocr": True,
        "details": {"dom": {"selector_syntax": "css"}},
    }
    captured: dict[str, Any] = {}

    async def _fake_run_node(_self, invocation_state, _input_map):  # pyright: ignore[reportMissingParameterType]
        captured["capabilities"] = invocation_state[CAPABILITIES_STATE_KEY]
        return NodeRun(output=ResponseToRPA(success=False))

    monkeypatch.setattr(Agent, "run_node", _fake_run_node, raising=True)

    headers = {"X-ROBOT-KEY": valid_robot_key_raw}
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers=headers
    ) as ws:
        ws.send_json(valid_recovery_payload | {"capabilities": capabilities})
        message = ws.receive_json()

    assert message["type"] == "done"
    recovery_context = session.exec(
        select(RecoveryContext).where(
            RecoveryContext.robot_exception_id == UUID(message["id"])
        )
    ).one()
    assert recovery_context.capabilities == capabilities
    assert captured["capabilities"] == capabilities


def test_recovery_ws_accepts_recovery_task_and_does_not_pass_it_to_the_agent(
    client: TestClient,
    valid_robot_key_raw: str,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    monkeypatch,  # pyright: ignore[reportMissingParameterType]
):
    calls: list[dict[str, Any]] = []

    async def _fake_run_node(_self, invocation_state, _input_map):  # pyright: ignore[reportMissingParameterType]
        inputs = invocation_state[AGENT_GRAPH_CONTEXT_KEY].recovery_inputs
        calls.append({"invocation_state": invocation_state, "inputs": inputs})
        return NodeRun(output=ResponseToRPA(success=False))

    monkeypatch.setattr(Agent, "run_node", _fake_run_node, raising=True)

    headers = {"X-ROBOT-KEY": valid_robot_key_raw}
    messages: list[dict[str, Any]] = []
    for payload in (
        valid_recovery_payload,
        valid_recovery_payload | {"recovery_task": "Approve the request"},
    ):
        with client.websocket_connect(
            "/recovery/robot_exception/ws", headers=headers
        ) as ws:
            ws.send_json(payload)
            messages.append(ws.receive_json())

    assert [message["type"] for message in messages] == ["done", "done"]
    without_task, with_task = calls
    # Ignored for now: the gateway gets exactly what it gets without the field.
    assert with_task["inputs"] == without_task["inputs"]
    assert "recovery_task" not in with_task["inputs"]
    assert "Approve the request" not in repr(with_task["inputs"])
    assert "Approve the request" not in repr(with_task["invocation_state"])


def test_recovery_ws_ingest_rejects_invalid_payload(
    session: Session,
    client: TestClient,
    valid_robot_key_raw: str,
):
    # No gateway agent: the payload is validated first, so a malformed payload
    # is reported as such even on a misconfigured server.
    headers = {"X-ROBOT-KEY": valid_robot_key_raw}
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers=headers
    ) as ws:
        ws.send_json({"task_name": "missing-graph"})
        message = ws.receive_json()
        closed = ws.receive()

    assert message["type"] == "error"
    assert message["code"] == "INVALID_RECOVERY_PAYLOAD"
    assert closed["code"] == 1003
    assert session.exec(select(RobotException)).first() is None


def test_recovery_ws_ingest_rejects_non_json_payload(
    client: TestClient,
    valid_robot_key_raw: str,
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
):
    headers = {"X-ROBOT-KEY": valid_robot_key_raw}
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers=headers
    ) as ws:
        ws.send_text("{not json")
        message = ws.receive_json()
        closed = ws.receive()

    assert message["type"] == "error"
    assert message["code"] == "INVALID_RECOVERY_PAYLOAD"
    assert closed["code"] == 1003


def test_missing_agent_graph_is_an_error_not_done(
    session: Session,
    client: TestClient,
    valid_robot_key_raw: str,
    valid_recovery_payload: dict[str, Any],
):
    headers = {"X-ROBOT-KEY": valid_robot_key_raw}
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers=headers
    ) as ws:
        ws.send_json(valid_recovery_payload)
        message = ws.receive_json()
        closed = ws.receive()

    assert message == {
        "type": "error",
        "code": "NO_AGENT_GRAPH",
        "content": "No enabled default recovery agent graph in the database.",
    }
    assert closed["code"] == 1011
    # No recovery was instantiated, so nothing was persisted.
    assert session.exec(select(RobotException)).first() is None
