"""``WS /evals/recovery/ws`` ends an unexpected failure the way a recovery does."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import routers.evals as evals_module
from database.agents.models import Agent
from tests.unit.fixtures.robot_ws_fixtures import receive_skipping_pings


# A ValueError too: one raised by the run is not the robot's bad request.
@pytest.mark.parametrize("error", [RuntimeError("model down"), ValueError("bad state")])
def test_an_exception_during_the_eval_is_an_internal_error(
    client: TestClient,
    make_robot_key,  # pyright: ignore[reportMissingParameterType]
    eval_gui_agent: Agent,
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    key_raw = "test-robot-key-eval-error-0001"
    _ = make_robot_key(name="Eval error", enabled=True, key_raw=key_raw)

    async def _recovery(*_: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(evals_module, "_run_standalone_recovery", _recovery)

    with client.websocket_connect(
        "/evals/recovery/ws",
        headers={"X-ROBOT-KEY": key_raw, "X-AGENT-ID": str(eval_gui_agent.id)},
    ) as ws:
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
