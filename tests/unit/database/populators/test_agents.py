"""The agent populator: seeded agents get their types, and their types' rules run."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlmodel import Session

from database.agents.behaviours import is_gui_agent
from database.agents.behaviours.gui.parsers import load_parser
from database.agents.models import Agent, AgentType
from database.populators.agents import _create_agent  # pyright: ignore[reportPrivateUsage]
from database.provider.models import Router


def test_the_seeds_give_each_gui_agent_its_type(
    seeded_agents: dict[str, Agent],
) -> None:
    uitars = seeded_agents["UI Standalone UITARS"]
    assert uitars.type == AgentType.TextGuiAgent
    assert uitars.parser_path is not None
    _ = load_parser(uitars.parser_path)
    assert seeded_agents["Tooled GuiAgent"].type == AgentType.TooledGuiAgent
    assert len([a for a in seeded_agents.values() if is_gui_agent(a)]) == 2


def test_seeding_a_text_gui_agent_without_a_parser_fails(
    session: Session, make_router: Callable[..., Router]
) -> None:
    router = make_router(
        capabilities=[Router.Capability.TEXT, Router.Capability.VISION]
    )

    with pytest.raises(ValueError, match="needs a parser_path"):
        _ = _create_agent(
            session=session,
            name="Reader",
            description="d",
            prompt="{instruction} {actions}",
            arguments=[],
            output_model=None,
            router=router,
            agent_type=AgentType.TextGuiAgent,
        )


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
        (
            "[populate_agents]   'Clicker' ignores output_model: a GUI agent always "
            "reports a GuiRunReport."
        )
    ]
