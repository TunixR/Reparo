"""What the reference client puts in the payload it sends first."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

PAYLOAD: dict[str, Any] = {"task_name": "Approve", "model": "<process/>"}


@pytest.fixture
def runner_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # pyautogui needs a display; the payload doesn't.
    for name in ("pyautogui", "pyperclip"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    for name in [m for m in sys.modules if m.startswith("scripts.recovery_ws_client")]:
        monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("scripts.recovery_ws_client.runner")


def _runner(
    runner_module: ModuleType,
    *,
    eval_mode: bool,
    task: str,
    payload: dict[str, Any] = PAYLOAD,
) -> Any:
    from scripts.recovery_ws_client.config import ClientConfig

    config = ClientConfig(
        ws_url="ws://localhost",
        robot_key="key",
        agent_id="agent",
        payload_file=Path("payloads.json"),
        payload_id=0,
        recovery_task=task,
        eval=eval_mode,
    )
    return runner_module.RecoveryWsRunner(config, dict(payload))


def test_eval_mode_sends_the_recovery_task_in_the_payload(
    runner_module: ModuleType,
) -> None:
    runner = _runner(runner_module, eval_mode=True, task="Approve the request")

    assert runner.initial_payload() == PAYLOAD | {
        "contract_version": 2,
        "recovery_task": "Approve the request",
    }


@pytest.mark.parametrize(("eval_mode", "task"), [(False, "Approve"), (True, "")])
def test_payload_has_no_recovery_task_otherwise(
    runner_module: ModuleType, eval_mode: bool, task: str
) -> None:
    runner = _runner(runner_module, eval_mode=eval_mode, task=task)

    assert runner.initial_payload() == PAYLOAD | {"contract_version": 2}


@pytest.mark.parametrize(
    "catalogue_payload", [PAYLOAD, PAYLOAD | {"contract_version": 1}]
)
def test_payload_speaks_contract_version_2_whatever_the_catalogue_holds(
    runner_module: ModuleType, catalogue_payload: dict[str, Any]
) -> None:
    runner = _runner(runner_module, eval_mode=False, task="", payload=catalogue_payload)

    assert runner.initial_payload()["contract_version"] == 2
