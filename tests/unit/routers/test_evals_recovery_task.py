"""``WS /evals/recovery/ws`` takes the recovery task from the payload, and only from there.

It ends a turn it can't run the way a recovery does: an ``error`` with a code, and the code as
the close reason.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import routers.evals as evals_module
from database.agents.models import Agent
from tests.unit.fixtures.robot_ws_fixtures import receive_skipping_pings

WS_PATH = "/evals/recovery/ws"
RESULT = {"success": False}


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-eval-task-0001"
    _ = make_robot_key(name="Eval task", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def seen_tasks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    tasks: list[str] = []

    async def _recovery(
        _robot: Any, _payload: Any, recovery_task: str, *_: Any
    ) -> dict[str, Any]:
        tasks.append(recovery_task)
        return RESULT

    monkeypatch.setattr(evals_module, "_run_standalone_recovery", _recovery)
    return tasks


def _connect(client: TestClient, robot_key_raw: str, agent: Agent) -> Any:
    return client.websocket_connect(
        WS_PATH,
        headers={"X-ROBOT-KEY": robot_key_raw, "X-AGENT-ID": str(agent.id)},
    )


def test_eval_uses_the_payload_recovery_task(
    client: TestClient,
    robot_key_raw: str,
    eval_gui_agent: Agent,
    valid_recovery_payload: dict[str, Any],
    seen_tasks: list[str],
) -> None:
    with _connect(client, robot_key_raw, eval_gui_agent) as ws:
        ws.send_json(valid_recovery_payload | {"recovery_task": "Approve the request"})
        message = receive_skipping_pings(ws)

    assert message["type"] == "eval_result"
    assert message["content"] == RESULT
    assert seen_tasks == ["Approve the request"]


@pytest.mark.parametrize("payload_task", [None, "", "  "])
def test_eval_without_a_recovery_task_is_rejected(
    client: TestClient,
    robot_key_raw: str,
    eval_gui_agent: Agent,
    valid_recovery_payload: dict[str, Any],
    seen_tasks: list[str],
    payload_task: str | None,
) -> None:
    with _connect(client, robot_key_raw, eval_gui_agent) as ws:
        ws.send_json(valid_recovery_payload | {"recovery_task": payload_task})
        message = receive_skipping_pings(ws)
        closed = ws.receive()

    assert message == {
        "type": "error",
        "code": "VALIDATION_ERROR",
        "content": "recovery_task must be a non-empty string",
    }
    assert closed["code"] == 1003
    assert closed["reason"] == "VALIDATION_ERROR"
    assert seen_tasks == []


def test_eval_still_rejects_unknown_payload_keys(
    client: TestClient,
    robot_key_raw: str,
    eval_gui_agent: Agent,
    valid_recovery_payload: dict[str, Any],
    seen_tasks: list[str],
) -> None:
    with _connect(client, robot_key_raw, eval_gui_agent) as ws:
        ws.send_json(valid_recovery_payload | {"recovery_plan": "Approve"})
        message = receive_skipping_pings(ws)
        closed = ws.receive()

    assert message["type"] == "error"
    assert message["code"] == "VALIDATION_ERROR"
    assert "recovery_plan" in message["content"]
    # The validation text can pass the 123 bytes a close reason may hold; the code fits.
    assert closed["code"] == 1003
    assert closed["reason"] == "VALIDATION_ERROR"
    assert seen_tasks == []


# A ValueError too: one raised by the run is not the robot's bad request.
@pytest.mark.parametrize("error", [RuntimeError("model down"), ValueError("bad state")])
def test_an_exception_during_the_eval_is_an_internal_error(
    client: TestClient,
    robot_key_raw: str,
    eval_gui_agent: Agent,
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    async def _recovery(*_: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(evals_module, "_run_standalone_recovery", _recovery)

    with _connect(client, robot_key_raw, eval_gui_agent) as ws:
        ws.send_json(valid_recovery_payload | {"recovery_task": "Approve the request"})
        message = receive_skipping_pings(ws)
        closed = ws.receive()

    assert message == {
        "type": "error",
        "code": "INTERNAL_ERROR",
        "content": "Internal server error",
    }
    assert closed["code"] == 1011
    assert closed["reason"] == "INTERNAL_ERROR"
