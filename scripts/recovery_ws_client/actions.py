from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pyautogui
import pyperclip

from .keys import resolve_keys
from .models import ActionResult


def _single_key(key: str) -> str:
    # The parser turns "space" into " ", so a bare space is the space bar.
    name = key if key == " " else key.strip()
    if not name:
        raise ValueError("requires a key")
    return resolve_keys([name], pyautogui.isValidKey)[0]


def _center_to_pixels(start_box: Any) -> tuple[int, int]:
    if not isinstance(start_box, list):
        raise TypeError("start_box must be a list")
    if len(start_box) == 2:
        x1 = x2 = float(start_box[0])
        y1 = y2 = float(start_box[1])
    elif len(start_box) == 4:
        x1 = float(start_box[0])
        y1 = float(start_box[1])
        x2 = float(start_box[2])
        y2 = float(start_box[3])
    else:
        raise ValueError("start_box must have length 2 or 4")

    width, height = pyautogui.size()
    if x1 <= 1.0 and x2 <= 1.0:
        x = int(((x1 + x2) / 2.0) * width)
    else:
        x = int((x1 + x2) / 2.0)
    if y1 <= 1.0 and y2 <= 1.0:
        y = int(((y1 + y2) / 2.0) * height)
    else:
        y = int((y1 + y2) / 2.0)
    return x, y


def _drag_to_pixels(start_box: Any, end_box: Any) -> tuple[int, int, int, int]:
    sx, sy = _center_to_pixels(start_box)
    ex, ey = _center_to_pixels(end_box)
    return sx, sy, ex, ey


# Each handler performs one action type with that action's inputs.
ActionHandler = Callable[[dict[str, Any]], None]


def _start_point(inputs: dict[str, Any]) -> tuple[int, int]:
    return _center_to_pixels(inputs.get("start_box"))


def _left_click(inputs: dict[str, Any]) -> None:
    x, y = _start_point(inputs)
    pyautogui.click(x=x, y=y, button="left")


def _double_click(inputs: dict[str, Any]) -> None:
    x, y = _start_point(inputs)
    pyautogui.doubleClick(x=x, y=y, button="left")


def _right_click(inputs: dict[str, Any]) -> None:
    x, y = _start_point(inputs)
    pyautogui.click(x=x, y=y, button="right")


def _hover(inputs: dict[str, Any]) -> None:
    x, y = _start_point(inputs)
    pyautogui.moveTo(x=x, y=y)


def _drag(inputs: dict[str, Any]) -> None:
    sx, sy, ex, ey = _drag_to_pixels(inputs.get("start_box"), inputs.get("end_box"))
    pyautogui.moveTo(x=sx, y=sy)
    pyautogui.dragTo(x=ex, y=ey, duration=1.0)


def _scroll(inputs: dict[str, Any]) -> None:
    direction = str(inputs.get("direction", "")).lower()
    match direction:
        case d if d in ["up", "down"]:
            scroll_fn = pyautogui.scroll
        case d if d in ["left", "right"]:
            scroll_fn = pyautogui.hscroll
        case _:
            raise ValueError(
                f"Invalid scroll direction: {direction}. Must be 'up', 'down', 'left', or 'right'."
            )

    x, y = _start_point(inputs)
    pyautogui.moveTo(x=x, y=y)
    amount = 200 if direction in ["up", "left"] else -200
    scroll_fn(amount, x=x, y=y)


def _hotkey(inputs: dict[str, Any]) -> None:
    hotkey = str(inputs.get("hotkey") or inputs.get("key") or "")
    pyautogui.hotkey(*resolve_keys(hotkey, pyautogui.isValidKey))


def _input_key(inputs: dict[str, Any]) -> str:
    return _single_key(str(inputs.get("key") or inputs.get("press") or ""))


def _key_down(inputs: dict[str, Any]) -> None:
    pyautogui.keyDown(_input_key(inputs))


def _key_up(inputs: dict[str, Any]) -> None:
    pyautogui.keyUp(_input_key(inputs))


def _type(inputs: dict[str, Any]) -> None:
    content = str(inputs.get("content") or "")
    pyperclip.copy(content.rstrip("\n"))
    pyautogui.hotkey("ctrl", "v")
    if content.endswith("\n"):
        pyautogui.press("enter")


def _wait(inputs: dict[str, Any]) -> None:
    seconds = inputs.get("seconds", inputs.get("duration", inputs.get("time", 0)))
    try:
        seconds_float = float(seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("wait requires a numeric seconds/duration") from exc
    if seconds_float < 0:
        raise ValueError("wait seconds must be non-negative")
    time.sleep(seconds_float)


ACTION_HANDLERS: dict[str, ActionHandler] = {
    "click": _left_click,
    "left_single": _left_click,
    "left_double": _double_click,
    "right_single": _right_click,
    "hover": _hover,
    "drag": _drag,
    "select": _drag,
    "scroll": _scroll,
    "hotkey": _hotkey,
    "keydown": _key_down,
    "press": _key_down,
    "keyup": _key_up,
    "release": _key_up,
    "type": _type,
    "wait": _wait,
}


def _perform(action_type: str, action_inputs: Any, dry_run: bool) -> None:
    if not isinstance(action_inputs, dict):
        raise TypeError("action_inputs must be an object")
    if dry_run:
        return
    handler = ACTION_HANDLERS.get(action_type)
    if handler is None:
        raise ValueError(f"Unsupported action_type: {action_type}")
    handler(action_inputs)


def execute_action(action: dict[str, Any], dry_run: bool = False) -> ActionResult:
    action_type = str(action.get("action_type", "unknown"))
    try:
        _perform(action_type, action.get("action_inputs", {}), dry_run)
    except Exception as exc:  # noqa: BLE001
        return ActionResult(success=False, action_type=action_type, error=str(exc))
    return ActionResult(success=True, action_type=action_type)
