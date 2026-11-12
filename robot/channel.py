"""The robot's side of a recovery websocket, read by a single task.

Once the payload is in, the robot only speaks when asked: an action result or a
screenshot. The exception is ``abort``, which it can send at any time, including
while an LLM is thinking and nothing is reading the socket. Read straight from
the socket, an abort would sit in the buffer and be taken for the next action
result.

``RobotChannel`` owns the socket's reads after the payload. One reader task
routes ``{"type": "abort"}`` to an abort event and queues every other frame for
``receive_json()`` / ``receive_bytes()``, the calls the GUI tools make. Once the
robot aborts, those calls fail with ``AbortedByRobot``, and the router, which
races the gateway against the abort, cancels the run.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar

from fastapi import WebSocketDisconnect

if TYPE_CHECKING:
    from types import TracebackType

    from fastapi import WebSocket

T = TypeVar("T")

ABORT_MESSAGE_TYPE = "abort"
# A robot abort is a normal closure: the robot asked for it.
ABORT_CLOSE_CODE = 1000
# Bounds what a robot can make us store; telemetry bounds it further.
ABORT_CODE_MAX_LENGTH = 255
ABORT_DESCRIPTION_MAX_LENGTH = 4096

_END = object()


class RobotConnection(Protocol):
    """What the GUI tools need from the robot: a ``RobotChannel`` in a recovery.

    A bare ``WebSocket`` fits too, for the standalone eval agents.
    """

    async def send_json(self, data: Any) -> None: ...

    async def receive_json(self) -> Any: ...

    async def receive_bytes(self) -> bytes: ...


@dataclass(frozen=True)
class RobotAbort:
    """Why the robot gave up on the recovery, as it reported it."""

    error_code: str | None
    error_description: str | None


def _bounded(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def parse_abort(data: Any) -> RobotAbort | None:
    """The abort ``data`` describes, if it is an ``abort`` message."""
    if not isinstance(data, dict) or data.get("type") != ABORT_MESSAGE_TYPE:  # pyright: ignore[reportUnknownMemberType]
        return None
    return RobotAbort(
        error_code=_bounded(data.get("error_code"), ABORT_CODE_MAX_LENGTH),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        error_description=_bounded(
            data.get("error_description"),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            ABORT_DESCRIPTION_MAX_LENGTH,
        ),
    )


def _parse_abort_frame(text: str) -> RobotAbort | None:
    try:
        return parse_abort(json.loads(text))
    except json.JSONDecodeError:
        return None


class AbortedByRobot(WebSocketDisconnect):
    """The robot aborted the recovery.

    A ``WebSocketDisconnect``, so the code paths that already let a lost robot
    end a GUI task treat an abort the same way instead of handing it to the model.
    """

    def __init__(self, abort: RobotAbort) -> None:
        super().__init__(code=ABORT_CLOSE_CODE, reason=abort.error_code)
        self.abort = abort


class RobotChannel:
    """Reads a recovery websocket after its payload; use as an async context manager.

    It stands in for the socket wherever the GUI tools read from it, and passes
    ``send_json`` through.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._frames: asyncio.Queue[str | bytes | object] = asyncio.Queue()
        self._aborted = asyncio.Event()
        self._ended: BaseException | None = None
        self._reader: asyncio.Task[None] | None = None
        self.abort: RobotAbort | None = None

    async def __aenter__(self) -> Self:
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._reader is not None:
            _ = self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader

    async def _read(self) -> None:
        try:
            while True:
                message = await self._websocket.receive()
                if message["type"] == "websocket.disconnect":
                    self._end(
                        WebSocketDisconnect(
                            code=message.get("code", 1000),
                            reason=message.get("reason"),
                        )
                    )
                    return
                text = message.get("text")
                if text is not None:
                    abort = _parse_abort_frame(text)
                    if abort is not None:
                        self.abort = abort
                        self._end(AbortedByRobot(abort))
                        self._aborted.set()
                        return
                    self._frames.put_nowait(text)
                elif (data := message.get("bytes")) is not None:
                    self._frames.put_nowait(data)
        except Exception as exc:  # noqa: BLE001 - handed to whoever waits for a frame
            self._end(exc)

    def _end(self, error: BaseException) -> None:
        if self._ended is None:
            self._ended = error
            self._frames.put_nowait(_END)

    def _raise_ended(self) -> None:
        if self.abort is not None:
            raise AbortedByRobot(self.abort)
        raise self._ended or WebSocketDisconnect()

    async def _next(self) -> str | bytes:
        # After an abort nothing else the robot sent matters.
        if self.abort is not None:
            self._raise_ended()
        frame = await self._frames.get()
        if frame is _END:
            # Leave it for the next waiter.
            self._frames.put_nowait(_END)
            self._raise_ended()
        return frame  # pyright: ignore[reportReturnType] - only _END is not a frame

    async def receive_json(self) -> Any:
        """The robot's next frame, which must be JSON text."""
        frame = await self._next()
        if isinstance(frame, bytes):
            raise TypeError("Expected a JSON text frame from the robot, got binary.")
        return json.loads(frame)

    async def receive_bytes(self) -> bytes:
        """The robot's next frame, which must be binary."""
        frame = await self._next()
        if isinstance(frame, str):
            raise TypeError("Expected a binary frame from the robot, got text.")
        return frame

    async def send_json(self, data: Any) -> None:
        await self._websocket.send_json(data)

    async def run_until_aborted(self, awaitable: Awaitable[T]) -> T:
        """Await ``awaitable``, cancelling it if the robot aborts first.

        Raises:
            AbortedByRobot: the robot aborted before ``awaitable`` finished.
        """
        work = asyncio.ensure_future(awaitable)
        aborted = asyncio.create_task(self._aborted.wait())
        try:
            _ = await asyncio.wait({work, aborted}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            _ = aborted.cancel()
            if self.abort is not None or not work.done():
                _ = work.cancel()
                # Let the run unwind; its own outcome no longer matters.
                _ = await asyncio.gather(work, return_exceptions=True)
        if self.abort is not None:
            raise AbortedByRobot(self.abort)
        return work.result()
