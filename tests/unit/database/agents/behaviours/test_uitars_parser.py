"""The UI-TARS parser reads the GUI actions out of UI-TARS replies."""

from __future__ import annotations

from typing import Any

import pytest

from database.agents.behaviours.gui.parsers import (
    ActionParseError,
    ActionParser,
    ParsedAction,
    load_parser,
)
from database.agents.behaviours.gui.parsers.uitars import WAIT_SECONDS, UITarsParser
from database.agents.behaviours.gui.shared import (
    DEFAULT_GUI_ACTIONS,
    GuiActions,
    GuiActionSpec,
)
from tests.unit.shared.gui_fakes import UITARS_PARSER

PARSER = UITarsParser()

# Replies as UI-TARS writes them: a Thought, then one or more calls after "Action:".
REPLIES: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
    (
        "Thought: The Submit button is at the bottom.\n"
        "Action: click(point='<point>235 512</point>')",
        [("click", {"start_box": [235.0, 512.0]})],
    ),
    (
        "Thought: Open the menu.\n"
        "Action: click(start_box='<|box_start|>(235,512)<|box_end|>')",
        [("click", {"start_box": [235.0, 512.0]})],
    ),
    (
        "Action: left_double(start_box='(10,20,30,40)')",
        [("double_click", {"start_box": [10.0, 20.0, 30.0, 40.0]})],
    ),
    (
        "Action: right_single(point='<point>5 6</point>')",
        [("right_click", {"start_box": [5.0, 6.0]})],
    ),
    (
        "Action: drag(start_point='<point>1 2</point>', end_point='<point>3 4</point>')",
        [("drag", {"start_box": [1.0, 2.0], "end_box": [3.0, 4.0]})],
    ),
    (
        "Action: select(start_box='(1,2)', end_box='(3,4)')",
        [("drag", {"start_box": [1.0, 2.0], "end_box": [3.0, 4.0]})],
    ),
    ("Action: hotkey(key='Ctrl C')", [("hotkey", {"key": "ctrl c"})]),
    ("Action: hotkey(hotkey='ArrowLeft')", [("hotkey", {"key": "left"})]),
    ("Action: press(key='shift')", [("key_down", {"key": "shift"})]),
    ("Action: release(key='shift')", [("key_up", {"key": "shift"})]),
    (
        "Action: type(content='Invoice 42\\n')",
        [("type_text", {"content": "Invoice 42\n"})],
    ),
    (
        "Action: scroll(point='<point>500 300</point>', direction='down')",
        [("scroll", {"start_box": [500.0, 300.0], "direction": "down"})],
    ),
    ("Action: wait()", [("wait", {"seconds": WAIT_SECONDS})]),
    (
        "Thought: The form is saved.\nAction: finished(content='Saved the form.')[EOS]",
        [("finished", {"content": "Saved the form."})],
    ),
    (
        "Action: alter_variable(name='budget', value='700')",
        [("alter_variable", {"name": "budget", "value": "700"})],
    ),
    # Several calls, one blank line apart.
    (
        "Thought: Fill the field and submit.\n"
        "Action: click(point='<point>1 2</point>')\n\n"
        "type(content='42')\n\n"
        "hotkey(key='enter')",
        [
            ("click", {"start_box": [1.0, 2.0]}),
            ("type_text", {"content": "42"}),
            ("hotkey", {"key": "enter"}),
        ],
    ),
    # Unescaped quotes and a raw newline in typed text, as UI-TARS often writes them.
    (
        "Action: type(content='It's \"done\"\nnow')",
        [("type_text", {"content": 'It\'s "done"\nnow'})],
    ),
    # Only the last "Action:" counts, so a Thought may quote one.
    (
        "Thought: Earlier I wrote Action: wait(), but now I click.\n"
        "Action: click(point='<point>7 8</point>')",
        [("click", {"start_box": [7.0, 8.0]})],
    ),
]


def test_the_ui_tars_parser_implements_the_interface() -> None:
    assert isinstance(PARSER, ActionParser)
    parser = load_parser(UITARS_PARSER)
    assert isinstance(parser, UITarsParser)


@pytest.mark.parametrize(("reply", "expected"), REPLIES)
def test_reads_the_actions_of_a_reply(
    reply: str, expected: list[tuple[str, dict[str, Any]]]
) -> None:
    assert PARSER.parse(reply) == [
        ParsedAction(name, inputs) for name, inputs in expected
    ]


def test_every_action_it_reads_validates_as_its_gui_action() -> None:
    for reply, _ in REPLIES:
        for action in PARSER.parse(reply):
            spec = DEFAULT_GUI_ACTIONS.get(action.name)
            assert spec is not None, action.name
            _ = spec.inputs.model_validate(action.inputs)


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("Thought: I am not sure what to do.", "no 'Action:'"),
        ("Thought: Hm.\nAction:   ", "no action after 'Action:'"),
        ("Action: teleport(point='<point>1 2</point>')", "Unknown action 'teleport'"),
        ("Action: click(point='<point>1</point>')", "Can't read a point or box"),
        ("Action: click('<point>1 2</point>')", "Can't read the action"),
        ("Action: click(point=x)", "Can't read the arguments"),
        ("Action: do something", "Can't read the action"),
    ],
)
def test_a_reply_it_cant_read_raises_a_parse_error(reply: str, message: str) -> None:
    with pytest.raises(ActionParseError, match=message):
        _ = PARSER.parse(reply)


def test_describe_lists_each_action_in_ui_tars_syntax() -> None:
    description = PARSER.describe(DEFAULT_GUI_ACTIONS)

    lines = description.splitlines()
    assert lines[0] == (
        "click(point='<point>x1 y1</point>') # Left-click on a UI element."
    )
    assert len(lines) == len(DEFAULT_GUI_ACTIONS.names()) + 1
    assert "escape characters" in lines[-1]


def test_describe_rejects_an_action_it_has_no_syntax_for() -> None:
    click = DEFAULT_GUI_ACTIONS.get("click")
    assert click is not None
    renamed = GuiActions(
        GuiActionSpec("tap", click.description, click.inputs, to_robot=click.to_robot)
    )

    with pytest.raises(ValueError, match="no syntax for the action 'tap'"):
        _ = PARSER.describe(renamed)
