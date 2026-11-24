"""A GUI agent tells its model which coordinate space to answer in."""

from __future__ import annotations

import pytest

from robot.capabilities import CAPABILITIES_STATE_KEY
from tests.unit.shared.gui_fakes import (
    drive,
    fake_router,
    make_gui_agent,
    make_robot,
    patch_gui_io,
    scripted_model,
)
from tests.unit.shared.scripted_model import ToolCall

pytestmark = pytest.mark.asyncio


async def test_instruction_states_the_model_coordinate_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_gui_io(monkeypatch)
    # Screenshots are 64x48 (width x height); this backbone doubles both sides.
    gui_agent = make_gui_agent(
        ToolCall("finished", {"content": "done"}),
        router=fake_router(relative_dim=lambda h, w: [h * 2, w * 2]),
    )

    _ = await drive(gui_agent, make_robot())

    first_prompt = scripted_model(gui_agent).requests[0][0]["content"][0]["text"]
    assert "128x96 space" in first_prompt
    assert "x from 0 to 128" in first_prompt
    assert "y from 0 to 96" in first_prompt


async def test_refuses_a_robot_without_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gui_io(monkeypatch)
    gui_agent = make_gui_agent(ToolCall("finished", {"content": "done"}))

    with pytest.raises(ValueError, match="only dom"):
        _ = await drive(
            gui_agent,
            make_robot(),
            state={CAPABILITIES_STATE_KEY: {"modalities": ["dom"], "ocr": False}},
        )

    assert scripted_model(gui_agent).requests == []
