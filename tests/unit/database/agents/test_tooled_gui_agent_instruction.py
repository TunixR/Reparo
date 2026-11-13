"""TooledGuiAgent tells its model which coordinate space to answer in."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from agent_tools.image import CAPABILITIES_STATE_KEY
from database.agents.tooled_gui_agent import TooledGuiAgent
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall
from tests.unit.shared.tooled_gui_fakes import GUI_TASK_INPUT, make_robot, patch_gui_io

pytestmark = pytest.mark.asyncio


async def test_instruction_states_the_model_coordinate_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_gui_io(monkeypatch)
    model = ScriptedModel(ToolCall("finished", {"content": "done"}))
    # Screenshots are 64x48 (width x height); this backbone doubles both sides.
    router = SimpleNamespace(
        relative_dim=lambda h, w: [h * 2, w * 2],
        original_coords=lambda x, y, _w, _h: [x, y],
        model_name="scripted",
        get_model=lambda **_: model,
    )
    agent = SimpleNamespace(
        id=uuid4(), prompt="{instruction}", router=router, get_model=lambda: model
    )
    gui_agent = TooledGuiAgent(agent)  # pyright: ignore[reportArgumentType]

    await gui_agent.tooled_guiagent(  # pyright: ignore[reportCallIssue]
        **GUI_TASK_INPUT, override_invocation_state={"websocket": make_robot()}
    )

    first_prompt = model.requests[0][0]["content"][0]["text"]
    assert "128x96 space" in first_prompt
    assert "x from 0 to 128" in first_prompt
    assert "y from 0 to 96" in first_prompt


async def test_refuses_a_robot_without_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gui_io(monkeypatch)
    model = ScriptedModel(ToolCall("finished", {"content": "done"}))
    router = SimpleNamespace(
        relative_dim=lambda h, w: [h, w],
        original_coords=lambda x, y, _w, _h: [x, y],
        model_name="scripted",
        get_model=lambda **_: model,
    )
    agent = SimpleNamespace(
        id=uuid4(), prompt="{instruction}", router=router, get_model=lambda: model
    )
    gui_agent = TooledGuiAgent(agent)  # pyright: ignore[reportArgumentType]

    with pytest.raises(ValueError, match="only dom"):
        await gui_agent.tooled_guiagent(  # pyright: ignore[reportCallIssue]
            **GUI_TASK_INPUT,
            override_invocation_state={
                "websocket": make_robot(),
                CAPABILITIES_STATE_KEY: {"modalities": ["dom"], "ocr": False},
            },
        )

    assert model.requests == []
