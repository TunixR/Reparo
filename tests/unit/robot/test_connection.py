"""``RobotConnection``: one object owns a recovery websocket from accept to close."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from robot.channel import (
    ABORT_DESCRIPTION_MAX_LENGTH,
    PING_MESSAGE,
    AbortedByRobot,
    KeepAliveFailed,
    RobotAbort,
    RobotConnection,
    parse_abort,
)

ABORT = '{"type": "abort", "error_code": "ENVIRONMENT_FAILURE", "error_description": "page closed"}'
NO_PINGS = 3600.0


class FakeAsgiWebSocket:
    """Delivers ASGI messages the test pushes, like Starlette's ``receive()``."""

    def __init__(self, send_delay: float = 0.0) -> None:
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[Any] = []
        self.closed: list[tuple[int, str | None]] = []
        self.send_error: Exception | None = None
        self.send_delay = send_delay
        self.sending = 0
        self.max_concurrent_sends = 0

    def push_text(self, text: str) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": text})

    def push_bytes(self, data: bytes) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": data})

    def push_disconnect(self, code: int = 1001) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": code})

    async def receive(self) -> dict[str, Any]:
        return await self.inbox.get()

    async def send_json(self, data: Any) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sending += 1
        self.max_concurrent_sends = max(self.max_concurrent_sends, self.sending)
        await asyncio.sleep(self.send_delay)
        self.sending -= 1
        self.sent.append(data)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))


def _connection(ws: FakeAsgiWebSocket, interval: float = NO_PINGS) -> RobotConnection:
    return RobotConnection(ws, keep_alive_interval=interval)  # pyright: ignore[reportArgumentType]


async def _thinking(cancelled: asyncio.Event) -> str:
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        cancelled.set()
        raise
    return "finished"


# --- Frames ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_frames_are_delivered_in_order_by_kind() -> None:
    ws = FakeAsgiWebSocket()
    ws.push_text('{"task_name": "payload"}')
    ws.push_text('{"success": true}')
    ws.push_bytes(b"\xff\xd8jpeg")

    async with _connection(ws) as robot:
        assert await robot.receive_text() == '{"task_name": "payload"}'
        await robot.send_json({"type": "action"})
        assert await robot.receive_json() == {"success": True}
        assert await robot.receive_bytes() == b"\xff\xd8jpeg"

    assert ws.sent == [{"type": "action"}]


@pytest.mark.asyncio
async def test_wrong_frame_kind_is_a_type_error() -> None:
    ws = FakeAsgiWebSocket()
    ws.push_text('{"success": true}')
    ws.push_bytes(b"jpeg")

    async with _connection(ws) as robot:
        with pytest.raises(TypeError, match="binary"):
            _ = await robot.receive_bytes()
        with pytest.raises(TypeError, match="text"):
            _ = await robot.receive_json()


# --- Endings -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_instead_of_the_payload_fails_the_read() -> None:
    ws = FakeAsgiWebSocket()
    ws.push_text(ABORT)

    async with _connection(ws) as robot:
        with pytest.raises(AbortedByRobot) as exc_info:
            _ = await robot.receive_text()

    assert exc_info.value.abort == RobotAbort("ENVIRONMENT_FAILURE", "page closed")
    # Existing "robot went away" paths propagate it instead of handing it to the model.
    assert isinstance(exc_info.value, WebSocketDisconnect)
    assert robot.abort == exc_info.value.abort


@pytest.mark.asyncio
async def test_abort_fails_a_pending_wait_and_later_ones_behind_queued_frames() -> None:
    ws = FakeAsgiWebSocket()

    async with _connection(ws) as robot:
        waiting = asyncio.create_task(robot.receive_json())
        await asyncio.sleep(0)
        ws.push_text(ABORT)
        with pytest.raises(AbortedByRobot):
            _ = await waiting

        ws.push_bytes(b"late screenshot")
        await asyncio.sleep(0.01)
        with pytest.raises(AbortedByRobot):
            _ = await robot.receive_bytes()


@pytest.mark.asyncio
async def test_abort_cancels_the_running_work() -> None:
    ws = FakeAsgiWebSocket()
    cancelled = asyncio.Event()

    async with _connection(ws) as robot:
        ws.push_text(ABORT)
        with pytest.raises(AbortedByRobot):
            _ = await robot.run(_thinking(cancelled))

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_disconnect_cancels_the_running_work_at_once() -> None:
    ws = FakeAsgiWebSocket()
    cancelled = asyncio.Event()

    async with _connection(ws) as robot:
        ws.push_disconnect(code=1001)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            _ = await robot.run(_thinking(cancelled))

    assert not isinstance(exc_info.value, AbortedByRobot)
    assert exc_info.value.code == 1001
    assert cancelled.is_set()
    assert robot.abort is None


@pytest.mark.asyncio
async def test_failed_ping_ends_the_connection() -> None:
    ws = FakeAsgiWebSocket()
    ws.send_error = ValueError("broken")
    cancelled = asyncio.Event()

    async with _connection(ws, interval=0.01) as robot:
        with pytest.raises(KeepAliveFailed) as exc_info:
            _ = await robot.run(_thinking(cancelled))

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_ping_to_a_robot_that_went_away_is_a_disconnect() -> None:
    ws = FakeAsgiWebSocket()
    ws.send_error = WebSocketDisconnect(code=1006)

    async with _connection(ws, interval=0.01) as robot:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            _ = await robot.receive_json()

    assert not isinstance(exc_info.value, KeepAliveFailed)


@pytest.mark.asyncio
async def test_run_returns_the_result_when_the_connection_stays_up() -> None:
    ws = FakeAsgiWebSocket()

    async def quick() -> str:
        return "finished"

    async with _connection(ws) as robot:
        assert await robot.run(quick()) == "finished"


# --- Sending and closing -----------------------------------------------------


@pytest.mark.asyncio
async def test_pings_are_sent_and_serialised_with_other_sends() -> None:
    ws = FakeAsgiWebSocket(send_delay=0.01)

    async with _connection(ws, interval=0.005) as robot:
        for _ in range(5):
            await robot.send_json({"type": "action"})

    assert PING_MESSAGE in ws.sent
    assert ws.max_concurrent_sends == 1


@pytest.mark.asyncio
async def test_close_is_idempotent_and_happens_on_exit() -> None:
    ws = FakeAsgiWebSocket()
    async with _connection(ws) as robot:
        await robot.close(1003, "INVALID")
        await robot.close()
    assert ws.closed == [(1003, "INVALID")]

    ws = FakeAsgiWebSocket()
    async with _connection(ws):
        pass
    assert ws.closed == [(1000, None)]

    ws = FakeAsgiWebSocket()
    with pytest.raises(RuntimeError):
        async with _connection(ws):
            raise RuntimeError("boom")
    assert ws.closed == [(1011, None)]


def test_parse_abort_bounds_the_reason_and_ignores_other_messages() -> None:
    abort = parse_abort(
        {"type": "abort", "error_code": "X", "error_description": "d" * 10_000}
    )
    assert abort is not None
    assert abort.error_code == "X"
    assert len(abort.error_description or "") == ABORT_DESCRIPTION_MAX_LENGTH

    assert parse_abort({"type": "abort"}) == RobotAbort(None, None)
    assert parse_abort({"success": True}) is None
    assert parse_abort(["abort"]) is None
