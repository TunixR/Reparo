"""What the reference client's command line runs and prints."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

PAYLOAD = {"task_name": "Approve", "ui_log": [], "errored_act": "click"}


class FakeRunner:
    """Stands in for the websocket run, ending the way a test sets it to."""

    created: ClassVar[list[FakeRunner]] = []
    status: ClassVar[int] = 0
    terminal_type: ClassVar[str] = "done"
    terminal_content: ClassVar[Any] = {"success": True}
    eval_result: ClassVar[dict[str, Any] | None] = None
    raises: ClassVar[BaseException | None] = None

    def __init__(self, config: Any, payload: dict[str, Any]) -> None:
        from scripts.recovery_ws_client.models import RunTranscript

        self.config = config
        self.payload = payload
        self.transcript = RunTranscript()
        self.transcript.summary.terminal_type = self.terminal_type
        self.transcript.summary.recovery_id = "rec-1"
        FakeRunner.created.append(self)

    async def run(self) -> int:
        if self.raises is not None:
            raise self.raises
        return self.status


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    for name in ("pyautogui", "pyperclip"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name in [m for m in sys.modules if m.startswith("scripts.recovery_ws")]:
        monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("scripts.recovery_ws_test_client")
    FakeRunner.created = []
    monkeypatch.setattr(module, "RecoveryWsRunner", FakeRunner)
    catalog = tmp_path / "payloads.json"
    _ = catalog.write_text(json.dumps([PAYLOAD]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return module


def _main(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, *extra: str, **runner: Any
) -> int:
    for name, value in runner.items():
        monkeypatch.setattr(FakeRunner, name, value)
    argv = ["client", "--ws-url", "ws://x", "--robot-key", "k", "--payload-id", "1"]
    monkeypatch.setattr(sys, "argv", [*argv, "--payload-file", "payloads.json", *extra])
    return client.main()


def test_a_done_run_prints_its_summary_and_content(
    client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(client, monkeypatch) == 0

    assert json.loads(capsys.readouterr().out) == {
        "terminal": "done",
        "actions_received": 0,
        "actions_success": 0,
        "actions_failed": 0,
        "recovery_id": "rec-1",
        "done": {"success": True},
    }
    (runner,) = FakeRunner.created
    assert runner.payload == PAYLOAD
    assert runner.config.recovery_task == ""


def test_an_error_run_prints_the_error(
    client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = _main(
        client, monkeypatch, status=1, terminal_type="error", terminal_content="bad"
    )

    assert status == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == "bad"
    assert "done" not in output


def test_an_eval_run_prints_and_logs_the_eval_result(
    client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    ground_truth = tmp_path / "gt.json"
    _ = ground_truth.write_text(
        json.dumps([{"ground_truth_levels": {"moderate": "Approve it"}}]),
        encoding="utf-8",
    )
    log = tmp_path / "logs" / "run.json"

    status = _main(
        client,
        monkeypatch,
        "--eval",
        "--ground-truth-file",
        str(ground_truth),
        "--log-json",
        str(log),
        eval_result={"score": 1},
    )

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["eval_result"] == {"score": 1}
    assert "done" not in output
    assert FakeRunner.created[0].config.recovery_task == "Approve it"
    logged = json.loads(log.read_text(encoding="utf-8"))
    assert logged["eval_result"] == {"score": 1}
    assert logged["summary"]["recovery_id"] == "rec-1"


def test_capabilities_replace_the_payloads(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = _main(client, monkeypatch, "--capabilities", '{"ocr": false}')

    assert FakeRunner.created[0].payload == PAYLOAD | {"capabilities": {"ocr": False}}


@pytest.mark.parametrize(
    "extra",
    [
        ("--capabilities", "{"),
        ("--capabilities", "[]"),
        ("--action-delay-seconds", "-1"),
    ],
)
def test_bad_arguments_exit_with_a_usage_error(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _ = _main(client, monkeypatch, *extra)
    assert excinfo.value.code == 2
    assert FakeRunner.created == []


def test_a_missing_payload_returns_2(
    client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(client, monkeypatch, "--payload-file", "missing.json") == 2

    assert "Payload error" in capsys.readouterr().err


def test_ctrl_c_returns_130(
    client: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _main(client, monkeypatch, raises=KeyboardInterrupt()) == 130


def test_an_unexpected_error_returns_1(
    client: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(client, monkeypatch, raises=RuntimeError("boom")) == 1

    assert "RuntimeError: boom" in capsys.readouterr().err
