# Recovery WS Test Client

A standalone WebSocket client that plays the robot's side of the recovery protocol, for local
desktop-driven end-to-end checks. It is not part of the pytest suite. For the protocol itself, see
the recovery lifecycle in [`ARCHITECTURE.md`](../../ARCHITECTURE.md).

The entry point is `scripts/recovery_ws_test_client.py`; the implementation is in this package.

## What It Does

- Connects to `/recovery/robot_exception/ws` with `X-ROBOT-KEY`.
- Sends a predefined payload selected from `scripts/data/payloads.json`.
- Responds to `type: "screenshot"` messages using `pyautogui.screenshot()` bytes.
- Executes `type: "action"` messages automatically using `pyautogui`.
- Enforces a non-blocking minimum delay between actions (default 1 second).
- Prints how the run ended: the `done.content` answer, or the `error` code and reason.
- Sends `abort` when you press Ctrl-C, so the server stops the recovery instead of waiting on a
  socket that went quiet.
- Writes optional transcript logs for run analysis.

## Payload Catalog

Payloads are stored in:

- `scripts/data/payloads.json`

Each entry has this shape:

```json
{
  "id": 1,
  "payload": {
    "task_name": "...",
    "platform": "...",
    "os": "...",
    "variables": {},
    "ui_log": [],
    "errored_act": {},
    "model": "...",
    "capabilities": {"modalities": ["vision"], "ocr": false}
  }
}
```

Use `--payload-id` to choose which payload to send. The payload is sent as stored, so a UI log entry
or `errored_act` that carries a `model_event_id` (the id of the task it performed in `model`) is
sent with it. `capabilities` is optional; without it the server assumes `vision` only. Use
`--capabilities` to send other capabilities without editing the catalog.

## Run

You need a running backend and an enabled robot key (create one with `POST /keys`, see `/scalar`).

```bash
uv run python -m scripts.recovery_ws_test_client \
  --ws-url ws://localhost:8000/recovery/robot_exception/ws \
  --robot-key "$X_ROBOT_KEY" \
  --payload-file scripts/data/payloads.json \
  --payload-id 1 \
  --action-delay-seconds 1 \
  --log-json tmp/recovery_ws_log.json
```

Optional flags:

- `--dry-run-actions`
- `--max-actions <n>`
- `--save-screenshots-dir <dir>`
- `--capabilities '<json>'`, e.g. `'{"modalities": ["dom", "vision"], "ocr": true}'`; replaces the
  payload's `capabilities`. A robot without `vision` can't be served by today's GUI agents, which
  makes this a quick way to see that refusal.

## Output

The client prints a JSON summary when the run ends. After a `done` it holds the answer:

```json
{
  "terminal": "done",
  "recovery_id": "...",
  "done": {
    "success": true,
    "process_finished": false,
    "continue_from_model_event_id": "act_4",
    "loop_iteration": [],
    "variables": {}
  }
}
```

After an `error` it holds `{"code": "...", "content": "..."}` instead. Ctrl-C sends
`{"type": "abort", "error_code": "USER_INTERRUPT", ...}` and exits with status 130. The exit status
is 0 for `done`, 1 for `error`, 2 when the server closed the socket, 3 on a client exception.

## Safety Notes

- Actions are performed on your active desktop.
- Keep `pyautogui` fail-safe enabled (default in runner).
- Avoid running this on a machine with unrelated foreground activity.

## Verification Checklist

1. Valid payload id runs and reaches websocket session.
2. Invalid payload id fails fast before connect.
3. Bad robot key is rejected by server.
4. Screenshot requests are answered with image bytes.
5. Action results are sent with success/failure shape.
6. Action execution spacing is at least 1 second.
7. `done` finalizes run with exit code 0 and prints its `content`.
8. Ctrl-C during a recovery sends `abort`; the server stores it on the exception and sends no `done`.
