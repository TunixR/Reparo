"""Both robot connections refuse any Contract version but 2 before validating the payload."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from database.agents.models import Agent
from database.keys.models import RobotKey
from database.logging.models import RobotException
from tests.unit.fixtures.robot_ws_fixtures import receive_skipping_pings

ROBOT_KEY_RAW = "test-robot-key-contract-0001"

Connect = Callable[[], AbstractContextManager[Any]]
Payload = dict[str, Any]


@pytest.fixture(params=["recovery", "eval"])
def connect(
    request: pytest.FixtureRequest,
    client: TestClient,
    make_robot_key: Callable[..., RobotKey],
) -> Connect:
    _ = make_robot_key(name="Contract", enabled=True, key_raw=ROBOT_KEY_RAW)
    headers = {"X-ROBOT-KEY": ROBOT_KEY_RAW}
    if request.param == "recovery":
        path = "/recovery/robot_exception/ws"
    else:
        agent: Agent = request.getfixturevalue("eval_gui_agent")
        path = "/evals/recovery/ws"
        headers["X-AGENT-ID"] = str(agent.id)
    return lambda: client.websocket_connect(path, headers=headers)


def _tasked(payload: Payload) -> Payload:
    """The payload as a valid eval payload too, so only its version can fail."""
    return payload | {"recovery_task": "Approve the request"}


def _versioned(version: object) -> Callable[[Payload], object]:
    return lambda payload: _tasked(payload) | {"contract_version": version}


def _unversioned(payload: Payload) -> object:
    return {k: v for k, v in _tasked(payload).items() if k != "contract_version"}


@pytest.mark.parametrize(
    ("build", "received"),
    [
        pytest.param(_unversioned, "none", id="missing"),
        pytest.param(_versioned(None), "none", id="null"),
        pytest.param(_versioned(1), "1", id="1"),
        pytest.param(_versioned(3), "3", id="3"),
        pytest.param(_versioned("2"), '"2"', id="string"),
        pytest.param(_versioned(2.0), "2.0", id="float"),
        pytest.param(_versioned(True), "true", id="bool"),
        # Not an object, so there is no version to read, even one that looks right.
        pytest.param(lambda payload: [_tasked(payload)], "none", id="non_object"),
        # The version is refused before the rest of the payload is looked at, so
        # neither the missing fields nor the eval's missing recovery task show.
        pytest.param(
            lambda payload: {"contract_version": 1, "task_name": payload["task_name"]},
            "1",
            id="wrong_version_on_invalid_payload",
        ),
    ],
)
def test_unsupported_contract_version_is_refused(
    session: Session,
    connect: Connect,
    valid_recovery_payload: Payload,
    build: Callable[[Payload], object],
    received: str,
) -> None:
    with connect() as ws:
        ws.send_json(build(valid_recovery_payload))
        message = receive_skipping_pings(ws)
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
        message = receive_skipping_pings(ws)
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
