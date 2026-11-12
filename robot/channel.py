"""A recovery websocket, owned by one object from accept to close.

The robot sends a payload, then only speaks when asked: an action result or a
screenshot. The exception is ``abort``, which it can send at any time, including
while an LLM is thinking and nothing is waiting for a frame. It can also just go
away, and the server pings it to keep the connection open.

``RobotConnection`` owns all of that. One reader task routes ``abort`` and
disconnects to the connection's ending, and queues every other frame for
``receive_text()`` / ``receive_json()`` / ``receive_bytes()``. A keep-alive task
pings. Every send goes through one lock. ``run()`` races the work (the gateway)
against the ending and cancels it as soon as the robot aborts, disconnects or a
ping fails. Routers never touch the raw socket after accepting it.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, TypeVar

from fastapi import WebSocketDisconnect

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from types import TracebackType

    from fastapi import WebSocket

T = TypeVar("T")

ABORT_MESSAGE_TYPE = "abort"
PING_MESSAGE = {"type": "ping"}
KEEP_ALIVE_INTERVAL = 10.0

# Websocket close codes (RFC 6455).
CLOSE_NORMAL = 1000
CLOSE_SERVER_ERROR = 1011
# A robot abort is a normal closure: the robot asked for it.
ABORT_CLOSE_CODE = CLOSE_NORMAL

# Bounds what a robot can make us store; telemetry bounds it further.
ABORT_CODE_MAX_LENGTH = 255
ABORT_DESCRIPTION_MAX_LENGTH = 4096

_END = object()


class RobotIO(Protocol):
    """What the GUI tools need from the robot: a ``RobotConnection`` in a recovery.

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
    # Skip decoding frames that can't be an abort, such as most payloads.
    if ABORT_MESSAGE_TYPE not in text:
        return None
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


class KeepAliveFailed(RuntimeError):
    """A ping failed for a reason other than the robot going away."""


class RobotConnection:
    """An accepted recovery websocket; use as an async context manager.

    Leaving the context stops the reader and the keep-alive, and closes the
    socket if nothing closed it yet: normally, or with 1011 on an exception.
    """

    def __init__(
        self, websocket: WebSocket, keep_alive_interval: float = KEEP_ALIVE_INTERVAL
    ) -> None:
        self._websocket = websocket
        self._keep_alive_interval = keep_alive_interval
        self._frames: asyncio.Queue[str | bytes | object] = asyncio.Queue()
        self._send_lock = asyncio.Lock()
        self._ended = asyncio.Event()
        self._ending: BaseException | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._closed = False

    @property
    def abort(self) -> RobotAbort | None:
        """The robot's abort, if that is how the connection ended."""
        return self._ending.abort if isinstance(self._ending, AbortedByRobot) else None

    # --- Lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> Self:
        self._tasks = [
            asyncio.create_task(self._read()),
            asyncio.create_task(self._keep_alive()),
        ]
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for task in self._tasks:
            _ = task.cancel()
        _ = await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.close(CLOSE_NORMAL if exc is None else CLOSE_SERVER_ERROR)

    def _end(self, ending: BaseException) -> None:
        """Record how the connection ended; only the first ending counts."""
        if self._ending is None:
            self._ending = ending
            self._frames.put_nowait(_END)
            self._ended.set()

    async def _read(self) -> None:
        try:
            while True:
                message = await self._websocket.receive()
                if message["type"] == "websocket.disconnect":
                    self._end(
                        WebSocketDisconnect(
                            code=message.get("code", CLOSE_NORMAL),
                            reason=message.get("reason"),
                        )
                    )
                    return
                text = message.get("text")
                if text is not None:
                    abort = _parse_abort_frame(text)
                    if abort is not None:
                        self._end(AbortedByRobot(abort))
                        return
                    self._frames.put_nowait(text)
                elif (data := message.get("bytes")) is not None:
                    self._frames.put_nowait(data)
        except Exception as exc:  # noqa: BLE001 - becomes the connection's ending
            self._end(exc)

    async def _keep_alive(self) -> None:
        while True:
            await asyncio.sleep(self._keep_alive_interval)
            try:
                await self.send_json(PING_MESSAGE)
            except WebSocketDisconnect as exc:
                self._end(exc)
                return
            except Exception as exc:  # noqa: BLE001 - becomes the connection's ending
                failure = KeepAliveFailed(
                    f"Keep-alive ping failed: {type(exc).__name__}"
                )
                failure.__cause__ = exc
                self._end(failure)
                return

    # --- Work ----------------------------------------------------------------

    async def run(self, awaitable: Awaitable[T]) -> T:
        """Await ``awaitable``, cancelling it if the connection ends first.

        Raises:
            AbortedByRobot: the robot aborted.
            WebSocketDisconnect: the robot went away.
            KeepAliveFailed: a ping failed.
        """
        work = asyncio.ensure_future(awaitable)
        ended = asyncio.create_task(self._ended.wait())
        try:
            _ = await asyncio.wait({work, ended}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            _ = ended.cancel()
            if self._ending is not None or not work.done():
                _ = work.cancel()
                # Let the work unwind; its own outcome no longer matters.
                _ = await asyncio.gather(work, return_exceptions=True)
        self._raise_if_ended()
        return work.result()

    def _raise_if_ended(self) -> None:
        if self._ending is not None:
            raise self._ending

    # --- Frames --------------------------------------------------------------

    async def _next(self) -> str | bytes:
        # Once the connection ended, nothing else the robot sent matters.
        self._raise_if_ended()
        frame = await self._frames.get()
        if frame is _END:
            self._raise_if_ended()
        return frame  # pyright: ignore[reportReturnType] - only _END is not a frame

    async def receive_text(self) -> str:
        """The robot's next frame, which must be text."""
        frame = await self._next()
        if isinstance(frame, bytes):
            raise TypeError("Expected a text frame from the robot, got binary.")
        return frame

    async def receive_json(self) -> Any:
        """The robot's next frame, which must be JSON text."""
        return json.loads(await self.receive_text())

    async def receive_bytes(self) -> bytes:
        """The robot's next frame, which must be binary."""
        frame = await self._next()
        if isinstance(frame, str):
            raise TypeError("Expected a binary frame from the robot, got text.")
        return frame

    async def send_json(self, data: Any) -> None:
        async with self._send_lock:
            await self._websocket.send_json(data)

    async def close(self, code: int = CLOSE_NORMAL, reason: str | None = None) -> None:
        """Close the socket once; later calls, or a robot already gone, are no-ops."""
        if self._closed:
            return
        self._closed = True
        async with self._send_lock:
            # The robot may already have closed its side.
            with suppress(RuntimeError, OSError, WebSocketDisconnect):
                await self._websocket.close(code=code, reason=reason)
