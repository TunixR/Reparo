"""How a recovery ends when the robot aborts or goes away: the gateway is cancelled and no ``done`` is sent."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlmodel import Session, select

import routers.evals as evals_module
import routers.recovery as recovery_module
from database.agents.models import Agent, AgentType
from database.logging.models import RobotException
from database.provider.models import Router
from robot.channel import AbortedByRobot, RobotConnection
from settings import TELEMETRY_MAX_TEXT_LENGTH

WS_PATH = "/recovery/robot_exception/ws"
LONG_DESCRIPTION = "page closed " * 50
ABORT = {
    "type": "abort",
    "error_code": "ENVIRONMENT_FAILURE",
    "error_description": LONG_DESCRIPTION,
}

GatewayCall = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-abort-0001"
    _ = make_robot_key(name="Abort", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(recovery_module, "tracer", provider.get_tracer("tests.abort"))
    try:
        yield exporter
    finally:
        provider.shutdown()


def _receive_skipping_pings(ws: Any) -> dict[str, Any]:
    while True:
        message = ws.receive()
        if message.get("text") and '"ping"' in message["text"]:
            continue
        return message


def _wait_for_turn(exporter: InMemorySpanExporter) -> ReadableSpan:
    """The finished ``recovery.turn`` span, once the handler is done."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        for span in exporter.get_finished_spans():
            if span.name == "recovery.turn":
                return span
        time.sleep(0.01)
    raise AssertionError("recovery.turn never finished")


def _gateway_cancelled_by_abort(
    run: Callable[[RobotConnection], Coroutine[Any, Any, None]],
) -> tuple[GatewayCall, dict[str, bool]]:
    """A gateway that runs ``run`` and records whether it was cancelled."""
    seen = {"cancelled": False, "finished": False}

    async def _call(
        _self: Agent, invocation_state: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        assert invocation_state is not None
        channel = invocation_state["websocket"]
        assert isinstance(channel, RobotConnection)
        try:
            await run(channel)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
        seen["finished"] = True
        return {"success": True}

    return _call, seen


def _assert_aborted(
    session: Session,
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    span_exporter: InMemorySpanExporter,
    seen: dict[str, bool],
    *,
    after_action: bool,
) -> None:
    with client.websocket_connect(
        WS_PATH, headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(payload)
        if after_action:
            action = ws.receive_json()
            assert action["type"] == "action"
        ws.send_json(ABORT)
        closed = _receive_skipping_pings(ws)

    # No ``done``: the next thing the robot sees is a normal closure.
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1000
    assert seen == {"cancelled": True, "finished": False}

    session.expire_all()
    exception = session.exec(select(RobotException)).one()
    assert exception.infered_success is False
    assert exception.abort_code == "ENVIRONMENT_FAILURE"
    assert exception.abort_description == LONG_DESCRIPTION

    turn = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "recovery.turn"
    )
    assert turn.attributes is not None
    assert turn.attributes["recovery.outcome"] == "robot_aborted"
    assert turn.attributes["recovery.success"] is False
    assert turn.status.status_code.name == "UNSET"
    event = next(event for event in turn.events if event.name == "robot_aborted")
    assert event.attributes is not None
    assert event.attributes["recovery.abort.code"] == "ENVIRONMENT_FAILURE"
    description = str(event.attributes["recovery.abort.description"])
    assert len(description) <= TELEMETRY_MAX_TEXT_LENGTH


def test_abort_mid_llm_turn_cancels_the_gateway(
    session: Session,
    client: TestClient,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    robot_key_raw: str,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _thinking(_robot: RobotConnection) -> None:
        # The LLM is thinking: nothing reads the socket.
        await asyncio.sleep(60)

    call, seen = _gateway_cancelled_by_abort(_thinking)
    monkeypatch.setattr(Agent, "__call__", call)

    _assert_aborted(
        session,
        client,
        robot_key_raw,
        valid_recovery_payload,
        span_exporter,
        seen,
        after_action=False,
    )


def test_abort_mid_action_is_not_read_as_the_action_result(
    session: Session,
    client: TestClient,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    robot_key_raw: str,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results: list[Any] = []

    async def _acting(channel: RobotConnection) -> None:
        await channel.send_json({"type": "action", "content": {"action_type": "click"}})
        try:
            results.append(await channel.receive_json())
        except AbortedByRobot as exc:
            # Even if a tool swallowed it, the router still cancels the run.
            results.append(exc)
        await asyncio.sleep(60)

    call, seen = _gateway_cancelled_by_abort(_acting)
    monkeypatch.setattr(Agent, "__call__", call)

    _assert_aborted(
        session,
        client,
        robot_key_raw,
        valid_recovery_payload,
        span_exporter,
        seen,
        after_action=True,
    )
    assert len(results) == 1
    assert isinstance(results[0], AbortedByRobot)


def test_abort_before_the_payload_just_closes(
    session: Session,
    client: TestClient,
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    robot_key_raw: str,
    span_exporter: InMemorySpanExporter,
) -> None:
    with client.websocket_connect(
        WS_PATH, headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(ABORT)
        closed = _receive_skipping_pings(ws)

    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1000
    assert session.exec(select(RobotException)).all() == []
    turn = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "recovery.turn"
    )
    assert turn.attributes is not None
    assert turn.attributes["recovery.outcome"] == "robot_aborted"


def test_disconnect_mid_llm_turn_cancels_the_gateway_and_is_recorded(
    session: Session,
    client: TestClient,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    robot_key_raw: str,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()

    async def _thinking(_robot: RobotConnection) -> None:
        started.set()
        await asyncio.sleep(60)

    call, seen = _gateway_cancelled_by_abort(_thinking)
    monkeypatch.setattr(Agent, "__call__", call)

    with client.websocket_connect(
        WS_PATH, headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(valid_recovery_payload)
        assert started.wait(timeout=5)
        # The robot goes away while the gateway is thinking.
        ws.close(code=1001)
        turn = _wait_for_turn(span_exporter)

    assert seen == {"cancelled": True, "finished": False}
    session.expire_all()
    exception = session.exec(select(RobotException)).one()
    assert exception.infered_success is False
    assert exception.abort_code is None

    assert turn.attributes is not None
    assert turn.attributes["recovery.outcome"] == "client_disconnected"


def test_eval_abort_mid_run_cancels_it_and_sends_no_result(
    session: Session,
    client: TestClient,
    valid_recovery_payload: dict[str, Any],
    make_agent: Callable[..., Agent],
    mock_router: Router,
    robot_key_raw: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gui_agent = make_agent(
        router=mock_router,
        name="GUI",
        description="GUI agent",
        prompt="prompt",
        response_model=None,
        type=AgentType.GuiAgent,
    )
    seen = {"cancelled": False}
    started = threading.Event()

    async def _thinking(*_: Any, **__: Any) -> dict[str, Any]:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            seen["cancelled"] = True
            raise
        return {}

    monkeypatch.setattr(evals_module, "_run_standalone_recovery", _thinking)
    # The agent lookup isn't under test (it passes the raw header to a UUID column).
    monkeypatch.setattr(evals_module, "_get_valid_agent", lambda *_: gui_agent)

    with client.websocket_connect(
        "/evals/recovery/ws",
        headers={"X-ROBOT-KEY": robot_key_raw, "X-AGENT-ID": str(gui_agent.id)},
    ) as ws:
        ws.send_json(valid_recovery_payload)
        ws.send_json({"recovery_task": "Approve the request"})
        assert started.wait(timeout=5)
        ws.send_json(ABORT)
        closed = _receive_skipping_pings(ws)

    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1000
    assert seen == {"cancelled": True}
    session.expire_all()
    exception = session.exec(select(RobotException)).one()
    assert exception.abort_code == "ENVIRONMENT_FAILURE"
    assert exception.infered_success is False
