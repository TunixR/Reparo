"""A TextGuiAgent reads its model's actions from text and runs them like any GUI action."""

from __future__ import annotations

from typing import Any

import pytest

from database.agents.behaviours import TextGuiAgent
from database.agents.behaviours.gui.parsers.uitars import UITarsParser
from database.agents.behaviours.gui.shared import DEFAULT_GUI_ACTIONS, GuiAction
from database.agents.behaviours.gui.text import MAX_PARSE_RETRIES
from tests.unit.shared.gui_fakes import (
    UITARS_PARSER,
    drive,
    make_gui_agent,
    make_robot,
    patch_gui_io,
    scripted_model,
)
from tests.unit.shared.scripted_model import Turn

pytestmark = pytest.mark.asyncio

CLICK = "Thought: Click Submit.\nAction: click(point='<point>10 20</point>')"
FINISHED = "Thought: Saved.\nAction: finished(content='Invoice filled')"


@pytest.fixture(autouse=True)
def _no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gui_io(monkeypatch)


def _text_agent(*turns: Turn) -> TextGuiAgent:
    return make_gui_agent(
        *turns,
        kind=TextGuiAgent,
        prompt="Actions:\n{actions}\nTask:\n{instruction}",
        parser_path=UITARS_PARSER,
    )


def _texts(message: dict[str, Any]) -> list[str]:
    return [block["text"] for block in message["content"] if "text" in block]


async def test_runs_the_actions_it_reads_until_finished() -> None:
    gui_agent = _text_agent(CLICK, FINISHED)
    robot = make_robot()

    run = await drive(gui_agent, robot)

    assert robot.actions == [
        {
            "type": "action",
            "content": {
                "action_type": "left_single",
                "action_inputs": {"start_box": [10.0, 20.0]},
            },
        }
    ]
    report = run.report()
    assert report.goal_reached
    assert report.summary == "Invoice filled"
    assert report.actions == [GuiAction(action_type="left_single", success=True)]
    # After the click the model sees the new screenshot.
    second = scripted_model(gui_agent).requests[1][-1]
    assert second["role"] == "user"
    assert any("image" in block for block in second["content"])


async def test_the_prompt_lists_the_actions_in_the_parsers_syntax() -> None:
    gui_agent = _text_agent(FINISHED)

    _ = await drive(gui_agent, make_robot())

    [first_text] = _texts(scripted_model(gui_agent).requests[0][0])
    assert UITarsParser().describe(DEFAULT_GUI_ACTIONS) in first_text
    assert "Task: Fill the invoice" in first_text


async def test_an_unreadable_reply_is_answered_with_the_error_and_retried() -> None:
    gui_agent = _text_agent("I think I should click.", CLICK, FINISHED)
    robot = make_robot()

    run = await drive(gui_agent, robot)

    assert run.report().goal_reached
    assert len(robot.actions) == 1
    retry = scripted_model(gui_agent).requests[1][-1]
    [text] = _texts(retry)
    assert "could not be read: The reply has no 'Action:'." in text


async def test_the_run_stops_after_too_many_unreadable_replies() -> None:
    gui_agent = _text_agent("No action here.")
    robot = make_robot()

    run = await drive(gui_agent, robot)

    assert scripted_model(gui_agent).calls == 1 + MAX_PARSE_RETRIES
    assert robot.actions == []
    assert run.telemetry.outcome == "unparseable_response"
    report = run.report()
    assert not report.goal_reached
    assert report.summary == "Stopped without calling finished."


async def test_a_failed_action_ends_the_reply_and_is_shown_to_the_model() -> None:
    two_clicks = (
        "Action: click(point='<point>1 2</point>')\n\nclick(point='<point>3 4</point>')"
    )
    gui_agent = _text_agent(two_clicks, FINISHED)
    robot = make_robot(action_success=False)

    run = await drive(gui_agent, robot)

    # The second click of the reply is not sent once the first failed.
    assert len(robot.actions) == 1
    assert run.report().actions == [GuiAction(action_type="left_single", success=False)]
    [feedback] = _texts(scripted_model(gui_agent).requests[1][-1])
    assert feedback.startswith("Action 'left_single' failed on client side.")


async def test_an_invalid_action_is_shown_to_the_model_and_not_sent() -> None:
    gui_agent = _text_agent(
        "Action: scroll(point='(1,2)', direction='sideways')", FINISHED
    )
    robot = make_robot()

    run = await drive(gui_agent, robot)

    assert robot.actions == []
    assert run.report().goal_reached
    [feedback] = _texts(scripted_model(gui_agent).requests[1][-1])
    assert feedback.startswith("Invalid inputs for 'scroll':")
