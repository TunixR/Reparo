"""The UI-TARS parser: a ``Thought``, then ``Action: name(arg='value', ...)`` calls.

UI-TARS writes points as ``<point>x y</point>``, ``(x,y)`` or
``<|box_start|>(x,y)<|box_end|>``, in the space of its resized screenshot (the
router's ``relative_dim``). The coordinates are kept as written; the GUI action
maps them back through the router. Several calls are separated by a blank line.
"""

from __future__ import annotations

import ast
import re
from typing import Any, ClassVar

from database.agents.behaviours.gui.parsers import ActionParseError, ParsedAction
from database.agents.behaviours.gui.shared import GuiActions

_ACTION_NAMES: dict[str, str] = {
    "click": "click",
    "left_single": "click",
    "left_double": "double_click",
    "right_single": "right_click",
    "hover": "hover",
    "drag": "drag",
    "select": "drag",
    "hotkey": "hotkey",
    "press": "key_down",
    "keydown": "key_down",
    "release": "key_up",
    "keyup": "key_up",
    "type": "type_text",
    "scroll": "scroll",
    "wait": "wait",
    "finished": "finished",
    "alter_variable": "alter_variable",
}
"""UI-TARS action names, and the GUI action each one is."""

_START_KEYS = frozenset({"point", "start_point", "start_box"})
_END_KEYS = frozenset({"end_point", "end_box"})
_KEY_KEYS = frozenset({"key", "hotkey"})
_KEY_NAMES = {
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
}

_CALL_SEPARATOR = re.compile(r"\)\s*\n\s*\n")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CONTENT_CALL = re.compile(r"(\w+)\(content='(.*)'\)", re.DOTALL)

WAIT_SECONDS = 5.0
"""How long ``wait()`` waits; UI-TARS gives no duration."""


class UITarsParser:
    SYNTAX: ClassVar[dict[str, str]] = {
        "click": "click(point='<point>x1 y1</point>')",
        "double_click": "left_double(point='<point>x1 y1</point>')",
        "right_click": "right_single(point='<point>x1 y1</point>')",
        "hover": "hover(point='<point>x1 y1</point>')",
        "type_text": "type(content='xxx')",
        "hotkey": "hotkey(key='ctrl c')",
        "key_down": "press(key='shift')",
        "key_up": "release(key='shift')",
        "scroll": "scroll(point='<point>x1 y1</point>', direction='down or up or right or left')",
        "drag": "drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')",
        "wait": f"wait() # Sleeps for {WAIT_SECONDS:g}s.",
        "finished": "finished(content='xxx')",
        "alter_variable": "alter_variable(name='xxx', value='xxx')",
    }
    """Each GUI action as UI-TARS writes it."""

    def describe(self, actions: GuiActions) -> str:
        lines: list[str] = []
        for spec in actions:
            syntax = self.SYNTAX.get(spec.name)
            if syntax is None:
                raise ValueError(
                    f"The UI-TARS parser has no syntax for the action '{spec.name}'."
                )
            lines.append(f"{syntax} # {spec.description}")
        lines.append(
            "Use escape characters \\', \\\" and \\n in content and value, so they "
            "parse as Python strings."
        )
        return "\n".join(lines)

    def parse(self, reply: str) -> list[ParsedAction]:
        text = reply.replace("[EOS]", "").strip()
        _, separator, actions = text.rpartition("Action:")
        if not separator:
            raise ActionParseError("The reply has no 'Action:'.")
        calls = [c.strip() for c in _CALL_SEPARATOR.split(actions.strip())]
        parsed = [self._action(c if c.endswith(")") else f"{c})") for c in calls if c]
        if not parsed:
            raise ActionParseError("The reply has no action after 'Action:'.")
        return parsed

    def _action(self, call: str) -> ParsedAction:
        name, arguments = _read_call(call)
        action = _ACTION_NAMES.get(name)
        if action is None:
            raise ActionParseError(f"Unknown action '{name}' in: {call}")
        inputs: dict[str, Any] = {}
        for key, value in arguments.items():
            if key in _START_KEYS:
                inputs["start_box"] = _coordinates(value, call)
            elif key in _END_KEYS:
                inputs["end_box"] = _coordinates(value, call)
            elif key in _KEY_KEYS:
                inputs["key"] = " ".join(
                    _KEY_NAMES.get(k, k) for k in str(value).lower().split()
                )
            else:
                inputs[key] = value
        if action == "wait":
            inputs.setdefault("seconds", WAIT_SECONDS)
        return ParsedAction(action, inputs)


def _read_call(call: str) -> tuple[str, dict[str, Any]]:
    """The name and keyword arguments of one ``name(key='value', ...)`` call."""
    # A newline inside a string would end it; as an escape, it parses.
    try:
        node = ast.parse(call.replace("\n", "\\n"), mode="eval").body
    except SyntaxError:
        node = None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
        arguments: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
                raise ActionParseError(f"Can't read the arguments of: {call}")
            arguments[keyword.arg] = keyword.value.value
        return node.func.id, arguments
    # Text with unescaped quotes, as UI-TARS often types it.
    if match := _CONTENT_CALL.fullmatch(call):
        content = match[2].replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n")
        return match[1], {"content": content}
    raise ActionParseError(f"Can't read the action: {call}")


def _coordinates(value: Any, call: str) -> list[float]:
    numbers = [float(n) for n in _NUMBER.findall(str(value))]
    if len(numbers) not in (2, 4):
        raise ActionParseError(f"Can't read a point or box from {value!r} in: {call}")
    return numbers
