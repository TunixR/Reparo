from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

import routers.evals as evals_module
from database.agents.models import Agent, AgentType
from database.logging.models import RobotException
from database.provider.models import Router


@pytest.mark.parametrize("success", [True, False])
def test_eval_stores_the_success_it_sends(
    client: TestClient,
    session: Session,
    make_robot_key,  # pyright: ignore[reportMissingParameterType]
    make_agent,  # pyright: ignore[reportMissingParameterType]
    mock_router: Router,
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    success: bool,
) -> None:
    key_raw = "test-robot-key-eval-success-0001"
    _ = make_robot_key(name="Eval success", enabled=True, key_raw=key_raw)
    agent: Agent = make_agent(
        router=mock_router,
        name="GUI",
        description="GUI agent",
        prompt="prompt",
        type=AgentType.GuiAgent,
    )
    monkeypatch.setattr(evals_module, "_get_valid_agent", lambda *_: agent)

    async def _recovery(*_: Any) -> dict[str, Any]:
        return {"success": success}

    monkeypatch.setattr(evals_module, "_run_standalone_recovery", _recovery)

    with client.websocket_connect(
        "/evals/recovery/ws",
        headers={"X-ROBOT-KEY": key_raw, "X-AGENT-ID": str(agent.id)},
    ) as ws:
        ws.send_json(valid_recovery_payload | {"recovery_task": "Approve the request"})
        message = ws.receive_json()
        while message.get("type") == "ping":
            message = ws.receive_json()

    assert message["type"] == "eval_result"
    session.expire_all()
    exception = session.exec(select(RobotException)).one()
    assert exception.infered_success is success
