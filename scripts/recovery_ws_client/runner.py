from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

import pyautogui
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from templates.recovery_payload import CONTRACT_VERSION

from .actions import execute_action
from .config import ClientConfig
from .logging import write_transcript
from .models import RunTranscript, TranscriptEvent
from .protocol import message_type, parse_message
from .screenshots import capture_screenshot_bytes


class RecoveryWsRunner:
    def __init__(self, config: ClientConfig, payload: dict[str, Any]):
        self.config = config
        self.payload = payload
        self.transcript = RunTranscript()
        self.action_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.next_action_at = datetime.now(tz=UTC)
        self.actions_executed = 0
        self.eval_result: dict[str, Any] | None = None
        self.terminal_content: Any = None

    async def _action_worker(self, websocket) -> None:  # pyright: ignore[reportMissingParameterType]
        while not self.stop_event.is_set():
            try:
                message = await asyncio.wait_for(self.action_queue.get(), timeout=None)
            except TimeoutError:
                continue

            now = datetime.now(tz=UTC)
            remaining = (self.next_action_at - now).total_seconds()
            if remaining > 0:
                await asyncio.sleep(remaining)

            action = message.get("content")
            if not isinstance(action, dict):
                result_payload = {
                    "success": False,
                    "action_type": "unknown",
                    "error": "Invalid action content",
                }
            else:
                result = execute_action(action, dry_run=self.config.dry_run_actions)
                result_payload = result.as_payload()

                if result.success:
                    self.transcript.summary.actions_success += 1
                else:
                    self.transcript.summary.actions_failed += 1

            self.actions_executed += 1
            if (
                self.config.max_actions is not None
                and self.actions_executed >= self.config.max_actions
            ):
                self.stop_event.set()

            await websocket.send(json.dumps(result_payload))
            self.transcript.add(
                TranscriptEvent.create(
                    direction="outbound",
                    event_type="action_result",
                    payload=result_payload,
                )
            )

            self.next_action_at = datetime.now(tz=UTC) + timedelta(
                seconds=self.config.action_delay_seconds
            )

    async def _receive_until_terminal(self, websocket) -> None:  # pyright: ignore[reportMissingParameterType]
        """Answer the server's requests until it sends a terminal message."""
        while not self.stop_event.is_set():
            raw = await websocket.recv()
            if not isinstance(raw, str):
                continue

            message = parse_message(raw)
            msg_type = message_type(message)

            self.transcript.add(
                TranscriptEvent.create(
                    direction="inbound",
                    event_type=msg_type or "unknown",
                    payload=message,
                )
            )

            if msg_type == "screenshot":
                await self._send_screenshot(websocket)
            elif msg_type == "action":
                self.transcript.summary.actions_received += 1
                await self.action_queue.put(message)
            elif self._end_on_terminal(msg_type, message):
                self.stop_event.set()
                break

    async def _send_screenshot(self, websocket) -> None:  # pyright: ignore[reportMissingParameterType]
        screenshot = capture_screenshot_bytes(save_dir=self.config.save_screenshots_dir)
        await websocket.send(screenshot)
        self.transcript.add(
            TranscriptEvent.create(
                direction="outbound",
                event_type="screenshot_bytes",
                payload={"bytes": len(screenshot)},
            )
        )

    def _end_on_terminal(self, msg_type: str, message: dict[str, Any]) -> bool:
        """Record a terminal message; False when `msg_type` is not one (e.g. `ping`)."""
        terminal_handlers = {
            "done": self._record_done,
            "eval_result": self._record_eval_result,
            "error": self._record_error,
        }
        handler = terminal_handlers.get(msg_type)
        if handler is None:
            return False
        handler(message)
        return True

    def _record_done(self, message: dict[str, Any]) -> None:
        self.transcript.summary.terminal_type = "done"
        self.terminal_content = message.get("content")
        recovery_id = message.get("id")
        if isinstance(recovery_id, str):
            self.transcript.summary.recovery_id = recovery_id

    def _record_eval_result(self, message: dict[str, Any]) -> None:
        self.transcript.summary.terminal_type = "done"
        self.eval_result = message["content"]
        self.transcript.add(
            TranscriptEvent.create(
                direction="inbound",
                event_type="eval_result",
                payload=message,
            )
        )

    def _record_error(self, message: dict[str, Any]) -> None:
        self.transcript.summary.terminal_type = "error"
        self.terminal_content = {
            "code": message.get("code"),
            "content": message.get("content"),
        }

    async def _send_abort(self, websocket) -> None:  # pyright: ignore[reportMissingParameterType]
        abort = {
            "type": "abort",
            "error_code": "USER_INTERRUPT",
            "error_description": "Interrupted with Ctrl-C in the recovery test client",
        }
        self.transcript.summary.terminal_type = "aborted"
        self.transcript.add(
            TranscriptEvent.create(
                direction="outbound", event_type="abort", payload=abort
            )
        )
        # Best effort: the server may already be gone.
        with suppress(ConnectionClosed, TimeoutError):
            await asyncio.wait_for(websocket.send(json.dumps(abort)), timeout=5)
        self._write_log()

    def _write_log(self) -> None:
        if self.config.log_json is not None:
            write_transcript(self.config.log_json, self.transcript)

    def _record_failure(
        self, terminal_type: str, event_type: str, exc: Exception
    ) -> None:
        self.transcript.summary.terminal_type = terminal_type
        self.transcript.add(
            TranscriptEvent.create(
                direction="internal", event_type=event_type, payload=str(exc)
            )
        )
        self._write_log()

    def initial_payload(self) -> dict[str, Any]:
        """The recovery payload in Reparo's Contract version, carrying the
        recovery task in eval mode.

        The version replaces whatever the catalog holds, so stored payloads
        keep working when the contract moves on.
        """
        payload = self.payload | {"contract_version": CONTRACT_VERSION}
        if self.config.eval and self.config.recovery_task:
            payload["recovery_task"] = self.config.recovery_task
        return payload

    async def run(self) -> int:
        pyautogui.FAILSAFE = True

        try:
            async with connect(
                self.config.ws_url,
                additional_headers={
                    "X-ROBOT-KEY": self.config.robot_key,
                    "X-AGENT-ID": self.config.agent_id,
                },
                max_size=None,
            ) as websocket:
                await websocket.send(json.dumps(self.initial_payload()))
                initial_event: dict[str, Any] = {"payload_id": self.config.payload_id}
                if self.config.eval and self.config.recovery_task:
                    initial_event["recovery_task"] = self.config.recovery_task
                self.transcript.add(
                    TranscriptEvent.create(
                        direction="outbound",
                        event_type="initial_payload",
                        payload=initial_event,
                    )
                )

                worker = asyncio.create_task(self._action_worker(websocket))
                try:
                    await self._receive_until_terminal(websocket)
                except asyncio.CancelledError:
                    # Ctrl-C: tell the server to stop the recovery instead of just hanging up.
                    await self._send_abort(websocket)
                    raise
                finally:
                    self.stop_event.set()
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

            self._write_log()
        except ConnectionClosed as exc:
            self._record_failure("disconnect", "connection_closed", exc)
            return 2
        except Exception as exc:  # noqa: BLE001
            self._record_failure("exception", "exception", exc)
            return 3
        else:
            return 0 if self.transcript.summary.terminal_type == "done" else 1
