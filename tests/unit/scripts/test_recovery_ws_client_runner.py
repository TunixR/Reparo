"""How the reference client answers the server until the recovery ends."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosed


class FakeWebSocket:
    def __init__(self, inbound: list[str | bytes]) -> None:
        self.inbound = list(inbound)
        self.sent: list[str | bytes] = []

    async def recv(self) -> str | bytes:
        return self.inbound.pop(0)

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)


@pytest.fixture
def runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # pyautogui needs a display; the message loop doesn't.
    for name in ("pyautogui", "pyperclip"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name in [m for m in sys.modules if m.startswith("scripts.recovery_ws_client")]:
        monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("scripts.recovery_ws_client.runner")
    monkeypatch.setattr(module, "capture_screenshot_bytes", lambda save_dir: b"jpeg")
    return module


@pytest.fixture
def runner(runner_module: ModuleType) -> Any:
    from scripts.recovery_ws_client.config import ClientConfig

    config = ClientConfig(
        ws_url="ws://localhost",
        robot_key="key",
        agent_id="agent",
        payload_file=Path("payloads.json"),
        payload_id=1,
    )
    return runner_module.RecoveryWsRunner(config, {})


def _messages(*messages: dict[str, Any] | bytes) -> list[str | bytes]:
    return [m if isinstance(m, bytes) else json.dumps(m) for m in messages]


def _events(runner: Any) -> list[tuple[str, str]]:
    return [(e.direction, e.event_type) for e in runner.transcript.events]


@pytest.mark.asyncio
async def test_requests_are_answered_until_done(runner: Any) -> None:
    action = {"type": "action", "content": {"action_type": "click"}}
    websocket = FakeWebSocket(
        _messages(
            b"binary is skipped",
            {"type": "ping"},
            {"type": "screenshot"},
            action,
            {"type": "done", "id": "rec-1", "content": {"success": True}},
            {"type": "ping"},
        )
    )

    await runner._receive_until_terminal(websocket)

    assert websocket.sent == [b"jpeg"]
    assert runner.action_queue.get_nowait() == action
    assert _events(runner) == [
        ("inbound", "ping"),
        ("inbound", "screenshot"),
        ("outbound", "screenshot_bytes"),
        ("inbound", "action"),
        ("inbound", "done"),
    ]
    summary = runner.transcript.summary
    assert (summary.terminal_type, summary.recovery_id) == ("done", "rec-1")
    assert summary.actions_received == 1
    assert runner.terminal_content == {"success": True}
    assert runner.stop_event.is_set()
    assert len(websocket.inbound) == 1


@pytest.mark.asyncio
async def test_a_done_without_a_string_id_keeps_no_recovery_id(runner: Any) -> None:
    websocket = FakeWebSocket(_messages({"type": "done", "id": 4}))

    await runner._receive_until_terminal(websocket)

    assert runner.transcript.summary.recovery_id is None


@pytest.mark.asyncio
async def test_an_eval_result_ends_the_run_as_done(runner: Any) -> None:
    websocket = FakeWebSocket(_messages({"type": "eval_result", "content": {"s": 1}}))

    await runner._receive_until_terminal(websocket)

    assert runner.transcript.summary.terminal_type == "done"
    assert runner.eval_result == {"s": 1}
    assert _events(runner) == [("inbound", "eval_result"), ("inbound", "eval_result")]
    assert runner.stop_event.is_set()


@pytest.mark.asyncio
async def test_an_error_ends_the_run_with_its_code(runner: Any) -> None:
    websocket = FakeWebSocket(
        _messages({"type": "error", "code": "BAD", "content": "nope"}, {"type": "x"})
    )

    await runner._receive_until_terminal(websocket)

    assert runner.transcript.summary.terminal_type == "error"
    assert runner.terminal_content == {"code": "BAD", "content": "nope"}
    assert runner.stop_event.is_set()
    assert len(websocket.inbound) == 1


@pytest.mark.asyncio
async def test_unknown_messages_are_logged_and_ignored(runner: Any) -> None:
    websocket = FakeWebSocket(_messages({"other": 1}, {"type": "error"}))

    await runner._receive_until_terminal(websocket)

    assert _events(runner) == [("inbound", "unknown"), ("inbound", "error")]


def _connect_to(websocket: FakeWebSocket | BaseException) -> Any:
    @asynccontextmanager
    async def connect(*_: Any, **__: Any) -> AsyncIterator[FakeWebSocket]:
        if isinstance(websocket, BaseException):
            raise websocket
        yield websocket

    return connect


@pytest.mark.parametrize(
    ("inbound", "status"),
    [
        ({"type": "done", "content": {}}, 0),
        ({"type": "error", "code": "BAD"}, 1),
    ],
)
@pytest.mark.asyncio
async def test_a_run_exits_with_how_the_recovery_ended(
    runner_module: ModuleType,
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    inbound: dict[str, Any],
    status: int,
) -> None:
    websocket = FakeWebSocket(_messages(inbound))
    monkeypatch.setattr(runner_module, "connect", _connect_to(websocket))
    runner.config.log_json = tmp_path / "run.json"

    assert await runner.run() == status

    assert json.loads(websocket.sent[0])["contract_version"] == 2
    assert _events(runner)[0] == ("outbound", "initial_payload")
    assert runner.config.log_json.exists()


@pytest.mark.parametrize(
    ("error", "status", "terminal_type", "event_type"),
    [
        (ConnectionClosed(None, None), 2, "disconnect", "connection_closed"),
        (OSError("refused"), 3, "exception", "exception"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_run_is_logged_with_its_exit_code(
    runner_module: ModuleType,
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    status: int,
    terminal_type: str,
    event_type: str,
) -> None:
    monkeypatch.setattr(runner_module, "connect", _connect_to(error))
    runner.config.log_json = tmp_path / "run.json"

    assert await runner.run() == status

    assert runner.transcript.summary.terminal_type == terminal_type
    assert _events(runner) == [("internal", event_type)]
    assert runner.transcript.events[0].payload == str(error)
    assert runner.config.log_json.exists()
