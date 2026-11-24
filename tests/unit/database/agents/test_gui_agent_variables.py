"""alter_variable keeps the JSON type a Variable was sent with (#308)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.shared.gui_fakes import (
    drive,
    make_gui_agent,
    make_robot,
    patch_gui_io,
    scripted_model,
)
from tests.unit.shared.scripted_model import ToolCall

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gui_io(monkeypatch)


async def _alter(variables: dict[str, Any], value: Any) -> tuple[dict[str, Any], Any]:
    """Run a GUI agent that alters ``budget`` to ``value`` and finishes.

    Returns the variables handed back and the alter_variable result the model saw.
    """
    gui_agent = make_gui_agent(
        ToolCall("alter_variable", {"name": "budget", "value": value}),
        ToolCall("finished", {"content": "done"}),
    )
    run = await drive(gui_agent, make_robot(), variables=variables)
    tool_result = scripted_model(gui_agent).requests[1][-1]["content"][0]["toolResult"]
    handed_back = run.variables
    return handed_back, tool_result


async def test_a_number_given_as_a_json_string_is_stored_as_a_number() -> None:
    variables, result = await _alter({"budget": 1000}, "700")

    assert variables == {"budget": 700}
    assert result["status"] == "success"
    assert "to 700." in result["content"][0]["text"]


async def test_a_value_of_another_type_is_rejected_and_leaves_the_variable() -> None:
    variables, result = await _alter({"budget": 1000}, "seven hundred")

    assert variables == {"budget": 1000}
    assert result["status"] == "error"
    assert "expected int" in result["content"][0]["text"]


@pytest.mark.parametrize(
    ("held", "sent", "stored"),
    [
        (False, "true", True),
        ([1], "[1, 2]", [1, 2]),
        ({"a": 1}, '{"a": 2}', {"a": 2}),
        (1000.5, 700, 700.0),
    ],
)
async def test_a_value_of_the_held_type_is_stored_parsed(
    held: Any, sent: Any, stored: Any
) -> None:
    variables, _ = await _alter({"budget": held}, sent)

    assert variables == {"budget": stored}


@pytest.mark.parametrize(
    ("held", "sent"),
    [
        (1000, True),
        (1000, 700.5),
        (1000, "NaN"),
        (1000, "Infinity"),
        (True, 1),
        ([1], '{"a": 1}'),
        ({"a": 1}, [1]),
    ],
)
async def test_a_value_of_another_json_type_is_rejected(held: Any, sent: Any) -> None:
    variables, result = await _alter({"budget": held}, sent)

    assert variables == {"budget": held}
    assert result["status"] == "error"


@pytest.mark.parametrize("held", ["1000", None])
@pytest.mark.parametrize("sent", ["700", 700, [7], {"v": 7}, None])
async def test_a_string_or_null_variable_accepts_any_value(
    held: Any, sent: Any
) -> None:
    variables, result = await _alter({"budget": held}, sent)

    assert variables == {"budget": sent}
    assert result["status"] == "success"
