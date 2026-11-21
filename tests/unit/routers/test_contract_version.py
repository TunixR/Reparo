"""Both robot connections refuse any Contract version but 2 before reading the payload."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

import routers.evals as evals_module
from database.agents.models import Agent, AgentType
from database.logging.models import RobotException
from database.provider.models import Router

ROBOT_KEY_RAW = "test-robot-key-contract-0001"
MISSING = object()

Connect = Callable[[], AbstractContextManager[Any]]


@pytest.fixture(params=["recovery", "eval"])
def connect(
    request: pytest.FixtureRequest,
    client: TestClient,
    make_robot_key,  # pyright: ignore[reportMissingParameterType]
    make_agent,  # pyright: ignore[reportMissingParameterType]
    mock_router: Router,
    monkeypatch: pytest.MonkeyPatch,
) -> Connect:
    _ = make_robot_key(name="Contract", enabled=True, key_raw=ROBOT_KEY_RAW)
    headers = {"X-ROBOT-KEY": ROBOT_KEY_RAW}
    if request.param == "recovery":
        path = "/recovery/robot_exception/ws"
    else:
        agent: Agent = make_agent(
            router=mock_router,
            name="GUI",
            description="GUI agent",
            prompt="prompt",
            type=AgentType.GuiAgent,
        )
        # The agent lookup isn't under test (it passes the raw header to a UUID column).
        monkeypatch.setattr(evals_module, "_get_valid_agent", lambda *_: agent)
        path = "/evals/recovery/ws"
        headers["X-AGENT-ID"] = str(agent.id)
    return lambda: client.websocket_connect(path, headers=headers)


def _receive_skipping_pings(ws: Any) -> dict[str, Any]:
    while True:
        message = ws.receive_json()
        if message.get("type") != "ping":
            return message


@pytest.mark.parametrize(
    ("contract_version", "received"),
    [
        (MISSING, "none"),
        (None, "none"),
        (1, "1"),
        (3, "3"),
        ("2", '"2"'),
        (2.0, "2.0"),
        (True, "true"),
    ],
)
def test_unsupported_contract_version_is_refused(
    session: Session,
    connect: Connect,
    valid_recovery_payload: dict[str, Any],
    contract_version: object,
    received: str,
) -> None:
    payload = valid_recovery_payload | {"recovery_task": "Approve the request"}
    del payload["contract_version"]
    if contract_version is not MISSING:
        payload["contract_version"] = contract_version

    with connect() as ws:
        ws.send_json(payload)
        message = _receive_skipping_pings(ws)
        closed = ws.receive()

    assert message == {
        "type": "error",
        "code": "UNSUPPORTED_CONTRACT_VERSION",
        "content": f"Reparo speaks contract version 2; the payload sent {received}.",
    }
    assert closed["code"] == 1003
    assert closed["reason"] == "UNSUPPORTED_CONTRACT_VERSION"
    assert session.exec(select(RobotException)).first() is None


@pytest.mark.parametrize(
    ("payload", "received"),
    [
        # Not an object, so there is no version to read, even one that looks right.
        ([{"contract_version": 2}], "none"),
        # The version is refused before the rest of the payload is looked at, so
        # neither the missing fields nor the eval's missing recovery task show.
        ({"contract_version": 1, "task_name": "Approve"}, "1"),
    ],
    ids=["non_object", "wrong_version_on_invalid_payload"],
)
def test_contract_version_is_checked_before_the_payload(
    session: Session, connect: Connect, payload: object, received: str
) -> None:
    with connect() as ws:
        ws.send_json(payload)
        message = _receive_skipping_pings(ws)
        closed = ws.receive()

    assert message == {
        "type": "error",
        "code": "UNSUPPORTED_CONTRACT_VERSION",
        "content": f"Reparo speaks contract version 2; the payload sent {received}.",
    }
    assert closed["code"] == 1003
    assert closed["reason"] == "UNSUPPORTED_CONTRACT_VERSION"
    assert session.exec(select(RobotException)).first() is None


def test_non_json_payload_is_an_invalid_payload(
    session: Session, connect: Connect
) -> None:
    with connect() as ws:
        ws.send_text("{not json")
        message = _receive_skipping_pings(ws)
        closed = ws.receive()

    assert message["type"] == "error"
    assert message["code"] == "INVALID_RECOVERY_PAYLOAD"
    assert closed["code"] == 1003
    assert session.exec(select(RobotException)).first() is None


def test_the_api_reference_documents_the_payload_contract_version(
    client: TestClient,
) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]

    payload = schemas["RecoveryPayload"]
    assert payload["properties"]["contract_version"]["const"] == 2
    assert "contract_version" in payload["required"]
    # Its nested models are documented too, so every reference resolves.
    assert {"RecoveryUiLogEntry", "RecoveryErroredAct", "RecoveryCapabilities"} <= set(
        schemas
    )
