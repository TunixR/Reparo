"""``RobotChannel``: one reader routes ``abort`` and queues every other frame."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from robot.channel import (
    ABORT_DESCRIPTION_MAX_LENGTH,
    AbortedByRobot,
    RobotAbort,
    RobotChannel,
    parse_abort,
)

ABORT = '{"type": "abort", "error_code": "ENVIRONMENT_FAILURE", "error_description": "page closed"}'


class FakeAsgiWebSocket:
    """Delivers ASGI messages the test pushes, like Starlette's ``receive()``."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[Any] = []

    def push_text(self, text: str) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": text})

    def push_bytes(self, data: bytes) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "bytes": data})

    def push_disconnect(self, code: int = 1001) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": code})

    async def receive(self) -> dict[str, Any]:
        return await self.inbox.get()

    async def send_json(self, data: Any) -> None:
        self.sent.append(data)


@pytest.mark.asyncio
async def test_frames_are_delivered_in_order_by_kind() -> None:
    ws = FakeAsgiWebSocket()
    ws.push_text('{"success": true}')
    ws.push_bytes(b"\xff\xd8jpeg")

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        await channel.send_json({"type": "action"})
        assert await channel.receive_json() == {"success": True}
        assert await channel.receive_bytes() == b"\xff\xd8jpeg"

    assert ws.sent == [{"type": "action"}]


@pytest.mark.asyncio
async def test_wrong_frame_kind_is_a_type_error() -> None:
    ws = FakeAsgiWebSocket()
    ws.push_text('{"success": true}')

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="binary"):
            _ = await channel.receive_bytes()


@pytest.mark.asyncio
async def test_abort_fails_a_pending_wait() -> None:
    ws = FakeAsgiWebSocket()

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        waiting = asyncio.create_task(channel.receive_json())
        await asyncio.sleep(0)
        ws.push_text(ABORT)

        with pytest.raises(AbortedByRobot) as exc_info:
            _ = await waiting

    assert exc_info.value.abort == RobotAbort("ENVIRONMENT_FAILURE", "page closed")
    # Existing "robot went away" paths propagate it instead of handing it to the model.
    assert isinstance(exc_info.value, WebSocketDisconnect)
    assert channel.abort == exc_info.value.abort


@pytest.mark.asyncio
async def test_abort_is_not_read_as_an_action_result_even_behind_queued_frames() -> (
    None
):
    ws = FakeAsgiWebSocket()
    ws.push_bytes(b"late screenshot")
    ws.push_text(ABORT)

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        await asyncio.sleep(0.01)
        for _ in range(2):
            with pytest.raises(AbortedByRobot):
                _ = await channel.receive_bytes()


@pytest.mark.asyncio
async def test_disconnect_delivers_queued_frames_then_raises() -> None:
    ws = FakeAsgiWebSocket()
    ws.push_text('{"success": true}')
    ws.push_disconnect(code=1001)

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        assert await channel.receive_json() == {"success": True}
        with pytest.raises(WebSocketDisconnect) as exc_info:
            _ = await channel.receive_json()

    assert not isinstance(exc_info.value, AbortedByRobot)
    assert exc_info.value.code == 1001


@pytest.mark.asyncio
async def test_run_until_aborted_cancels_the_run_on_abort() -> None:
    ws = FakeAsgiWebSocket()
    cancelled = asyncio.Event()

    async def thinking() -> str:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "finished"

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        ws.push_text(ABORT)
        with pytest.raises(AbortedByRobot):
            _ = await channel.run_until_aborted(thinking())

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_run_until_aborted_returns_the_result_without_abort() -> None:
    ws = FakeAsgiWebSocket()

    async def quick() -> str:
        return "finished"

    async with RobotChannel(ws) as channel:  # pyright: ignore[reportArgumentType]
        assert await channel.run_until_aborted(quick()) == "finished"


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
