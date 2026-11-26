"""How the reference client turns an action message into desktop input."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

SCREEN = (1000, 500)
VALID_KEYS = {"ctrl", "shift", "alt", "v", "c", "enter", "a", " "}

Calls = list[tuple[str, tuple[Any, ...], dict[str, Any]]]


def _recorder(calls: Calls, name: str) -> Callable[..., None]:
    def record(*args: Any, **kwargs: Any) -> None:
        calls.append((name, args, kwargs))

    return record


@pytest.fixture
def calls() -> Calls:
    return []


@pytest.fixture
def execute(
    monkeypatch: pytest.MonkeyPatch, calls: Calls
) -> Callable[..., dict[str, Any]]:
    # pyautogui needs a display; a recording stand-in shows what would be sent.
    pyautogui = ModuleType("pyautogui")
    for name in (
        "click",
        "doubleClick",
        "moveTo",
        "dragTo",
        "scroll",
        "hscroll",
        "hotkey",
        "keyDown",
        "keyUp",
        "press",
    ):
        setattr(pyautogui, name, _recorder(calls, name))
    pyautogui.size = lambda: SCREEN  # pyright: ignore[reportAttributeAccessIssue]
    pyautogui.isValidKey = VALID_KEYS.__contains__  # pyright: ignore[reportAttributeAccessIssue]
    pyperclip = ModuleType("pyperclip")
    pyperclip.copy = _recorder(calls, "copy")  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    monkeypatch.setitem(sys.modules, "pyperclip", pyperclip)
    for name in [m for m in sys.modules if m.startswith("scripts.recovery_ws_client")]:
        monkeypatch.delitem(sys.modules, name)
    actions = importlib.import_module("scripts.recovery_ws_client.actions")
    monkeypatch.setattr(actions.time, "sleep", _recorder(calls, "sleep"))

    def run(action: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
        return actions.execute_action(action, dry_run=dry_run).as_payload()

    return run


@pytest.mark.parametrize(
    ("action_type", "inputs", "expected"),
    [
        (
            "click",
            {"start_box": [0.5, 0.5]},
            [("click", (), {"x": 500, "y": 250, "button": "left"})],
        ),
        (
            "left_single",
            {"start_box": [10, 20, 30, 40]},
            [("click", (), {"x": 20, "y": 30, "button": "left"})],
        ),
        (
            "left_double",
            {"start_box": [0.1, 0.2]},
            [("doubleClick", (), {"x": 100, "y": 100, "button": "left"})],
        ),
        (
            "right_single",
            {"start_box": [0.1, 0.2]},
            [("click", (), {"x": 100, "y": 100, "button": "right"})],
        ),
        ("hover", {"start_box": [0.1, 0.2]}, [("moveTo", (), {"x": 100, "y": 100})]),
        (
            "drag",
            {"start_box": [0.1, 0.2], "end_box": [0.5, 0.5]},
            [
                ("moveTo", (), {"x": 100, "y": 100}),
                ("dragTo", (), {"x": 500, "y": 250, "duration": 1.0}),
            ],
        ),
        (
            "select",
            {"start_box": [0.1, 0.2], "end_box": [0.5, 0.5]},
            [
                ("moveTo", (), {"x": 100, "y": 100}),
                ("dragTo", (), {"x": 500, "y": 250, "duration": 1.0}),
            ],
        ),
        (
            "scroll",
            {"direction": "Up", "start_box": [0.5, 0.5]},
            [
                ("moveTo", (), {"x": 500, "y": 250}),
                ("scroll", (200,), {"x": 500, "y": 250}),
            ],
        ),
        (
            "scroll",
            {"direction": "down", "start_box": [0.5, 0.5]},
            [
                ("moveTo", (), {"x": 500, "y": 250}),
                ("scroll", (-200,), {"x": 500, "y": 250}),
            ],
        ),
        (
            "scroll",
            {"direction": "left", "start_box": [0.5, 0.5]},
            [
                ("moveTo", (), {"x": 500, "y": 250}),
                ("hscroll", (200,), {"x": 500, "y": 250}),
            ],
        ),
        (
            "scroll",
            {"direction": "right", "start_box": [0.5, 0.5]},
            [
                ("moveTo", (), {"x": 500, "y": 250}),
                ("hscroll", (-200,), {"x": 500, "y": 250}),
            ],
        ),
        ("hotkey", {"hotkey": "ctrl c"}, [("hotkey", ("ctrl", "c"), {})]),
        ("hotkey", {"key": "ctrl a"}, [("hotkey", ("ctrl", "a"), {})]),
        ("keydown", {"key": "shift"}, [("keyDown", ("shift",), {})]),
        ("press", {"press": "alt"}, [("keyDown", ("alt",), {})]),
        ("keyup", {"key": " "}, [("keyUp", (" ",), {})]),
        ("release", {"press": "alt"}, [("keyUp", ("alt",), {})]),
        (
            "type",
            {"content": "hello"},
            [("copy", ("hello",), {}), ("hotkey", ("ctrl", "v"), {})],
        ),
        (
            "type",
            {"content": "hello\n"},
            [
                ("copy", ("hello",), {}),
                ("hotkey", ("ctrl", "v"), {}),
                ("press", ("enter",), {}),
            ],
        ),
        ("wait", {"seconds": 2}, [("sleep", (2.0,), {})]),
        ("wait", {"duration": "1.5"}, [("sleep", (1.5,), {})]),
        ("wait", {"time": 3}, [("sleep", (3.0,), {})]),
        ("wait", {}, [("sleep", (0.0,), {})]),
    ],
)
def test_an_action_drives_the_desktop(
    execute: Callable[..., dict[str, Any]],
    calls: Calls,
    action_type: str,
    inputs: dict[str, Any],
    expected: Calls,
) -> None:
    result = execute({"action_type": action_type, "action_inputs": inputs})

    assert result == {"success": True, "action_type": action_type}
    assert calls == expected


@pytest.mark.parametrize(
    ("action", "error"),
    [
        (
            {"action_type": "click", "action_inputs": []},
            "action_inputs must be an object",
        ),
        ({"action_type": "fly", "action_inputs": {}}, "Unsupported action_type: fly"),
        ({"action_inputs": {}}, "Unsupported action_type: unknown"),
        (
            {"action_type": "click", "action_inputs": {"start_box": "0.5"}},
            "start_box must be a list",
        ),
        (
            {"action_type": "click", "action_inputs": {"start_box": [1, 2, 3]}},
            "start_box must have length 2 or 4",
        ),
        (
            {"action_type": "scroll", "action_inputs": {"direction": "sideways"}},
            "Invalid scroll direction: sideways. Must be 'up', 'down', 'left', or 'right'.",
        ),
        (
            {"action_type": "keydown", "action_inputs": {}},
            "requires a key",
        ),
        (
            {"action_type": "wait", "action_inputs": {"seconds": "soon"}},
            "wait requires a numeric seconds/duration",
        ),
        (
            {"action_type": "wait", "action_inputs": {"seconds": -1}},
            "wait seconds must be non-negative",
        ),
    ],
)
def test_a_bad_action_fails_without_raising(
    execute: Callable[..., dict[str, Any]],
    calls: Calls,
    action: dict[str, Any],
    error: str,
) -> None:
    result = execute(action)

    assert result == {
        "success": False,
        "action_type": action.get("action_type", "unknown"),
        "error": error,
    }
    assert calls == []


def test_a_dry_run_touches_nothing_even_for_unknown_actions(
    execute: Callable[..., dict[str, Any]], calls: Calls
) -> None:
    result = execute({"action_type": "fly", "action_inputs": {}}, dry_run=True)

    assert result == {"success": True, "action_type": "fly"}
    assert calls == []


def test_a_dry_run_still_rejects_inputs_that_are_not_an_object(
    execute: Callable[..., dict[str, Any]],
) -> None:
    result = execute({"action_type": "click", "action_inputs": "x"}, dry_run=True)

    assert result["success"] is False
