"""How a `TextGuiAgent` reads actions out of its model's free-text replies.

A parser knows one model family's action syntax. It describes the GUI actions to
the model in that syntax, and turns a reply back into actions named and shaped as
in `DEFAULT_GUI_ACTIONS`, with coordinates left in the model's space. The agent
row names its parser by import path (``parser_path``, ``module:attribute``); the
attribute is a parser class, built without arguments, or a parser instance. Only
modules of this package can be named.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from database.agents.behaviours.gui.shared import GuiActions

PARSERS_PACKAGE = __name__
"""The package every parser module lives in."""


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """An action read from a reply: its name and raw inputs, as the GUI actions take them."""

    name: str
    inputs: dict[str, Any]


class ActionParseError(ValueError):
    """A reply holds no action the parser can read."""


@runtime_checkable
class ActionParser(Protocol):
    def describe(self, actions: GuiActions) -> str:
        """The action space for the prompt: every action of ``actions`` in this
        parser's syntax, with its description."""
        ...

    def parse(self, reply: str) -> Sequence[ParsedAction]:
        """The actions in ``reply``, in order. Raises `ActionParseError`."""
        ...


def load_parser(path: str) -> ActionParser:
    """The parser ``path`` names. Raises `ValueError` when it can't be used."""
    module_name, sep, attribute = path.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(
            f"parser_path '{path}' must name a parser as 'module:attribute'."
        )
    if not module_name.startswith(f"{PARSERS_PACKAGE}."):
        raise ValueError(
            f"parser_path '{path}' must name a module in {PARSERS_PACKAGE}."
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"parser_path '{path}': the module can't be imported ({exc})."
        ) from exc
    try:
        target = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(
            f"parser_path '{path}': {module_name} has no attribute '{attribute}'."
        ) from exc
    if isinstance(target, type):
        try:
            target = target()
        except TypeError as exc:
            raise ValueError(
                f"parser_path '{path}': the parser class can't be built without arguments ({exc})."
            ) from exc
    if not isinstance(target, ActionParser):
        raise ValueError(
            f"parser_path '{path}' doesn't implement the parser interface "
            "(describe and parse)."
        )
    return target
