"""The agent populator: the seeded agents pass their types' rules."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlmodel import Session

from database.agents.behaviours import is_gui_agent, validate_agent
from database.agents.behaviours.gui.parsers import load_parser
from database.agents.behaviours.gui.shared import GUI_ARGUMENTS
from database.agents.models import Agent, AgentType
from database.agents.specs import dump_fields
from database.populators.agents import _create_agent  # pyright: ignore[reportPrivateUsage]
from database.provider.models import Router


def test_the_seeded_agents_pass_their_types_rules(
    seeded_agents: dict[str, Agent],
) -> None:
    for agent in seeded_agents.values():
        _ = validate_agent(agent)

    uitars = seeded_agents["UI Standalone UITARS"]
    assert uitars.type == AgentType.TextGuiAgent
    assert uitars.parser_path is not None
    _ = load_parser(uitars.parser_path)
    assert seeded_agents["Tooled GuiAgent"].type == AgentType.TooledGuiAgent
    gui_agents = [a for a in seeded_agents.values() if is_gui_agent(a)]
    assert len(gui_agents) == 2
    for agent in gui_agents:
        assert dump_fields(agent.arguments) == dump_fields(GUI_ARGUMENTS)
        assert agent.input_type == Agent.InputType.IMAGETEXT


def test_only_the_ignored_fields_an_entry_sets_are_printed(
    session: Session,
    make_router: Callable[..., Router],
    capsys: pytest.CaptureFixture[str],
) -> None:
    router = make_router(
        capabilities=[Router.Capability.TEXT, Router.Capability.VISION]
    )

    _ = _create_agent(
        session=session,
        name="Clicker",
        description="d",
        prompt="{instruction}",
        arguments=[],
        output_model={"name": "Report", "description": "A report.", "fields": []},
        router=router,
        agent_type=AgentType.TooledGuiAgent,
        given={"name", "output_model"},
    )

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[populate_agents] Created agent 'Clicker'."
    assert lines[1:] == [
        "[populate_agents]   'Clicker' ignores output_model: a GUI agent always "
        "reports a GuiRunReport."
    ]
