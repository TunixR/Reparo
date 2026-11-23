"""Each agent type runs as its behaviour class, and its rules run in `validate_agent`."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from database.agents.behaviours import (
    AgentBehaviour,
    DefaultAgent,
    GuiAgent,
    TextGuiAgent,
    TooledGuiAgent,
    agent_types_of,
    behaviour_class,
    behaviour_of,
    is_gui_agent,
    validate_agent,
)
from database.agents.behaviours.gui.shared import GUI_ARGUMENTS
from database.agents.models import Agent, AgentType, SubAgent
from database.agents.specs import dump_fields
from database.provider.models import Router

UITARS_PARSER = "database.agents.behaviours.gui.parsers.uitars:UITarsParser"
TASK = {"name": "task", "type": "str", "description": "Task."}
REPORT = {"name": "Report", "description": "A report.", "fields": []}


@pytest.fixture
def vision_router(make_router: Callable[..., Router]) -> Router:
    return make_router(
        capabilities=[Router.Capability.TEXT, Router.Capability.VISION], persist=False
    )


def _agent(router: Router, type_: AgentType, **overrides: Any) -> Agent:
    """An in-memory agent of ``type_``; a TextGuiAgent gets the UI-TARS parser."""
    fields: dict[str, Any] = {
        "name": "An Agent",
        "description": "d",
        "prompt": "{instruction}",
        "input_type": Agent.InputType.TEXT,
        "parser_path": UITARS_PARSER if type_ is AgentType.TextGuiAgent else None,
    }
    return Agent(router=router, type=type_, **fields | overrides)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("agent_type", "cls"),
    [
        (AgentType.Agent, DefaultAgent),
        (AgentType.TooledGuiAgent, TooledGuiAgent),
        (AgentType.TextGuiAgent, TextGuiAgent),
    ],
)
def test_each_type_runs_as_its_class(
    vision_router: Router, agent_type: AgentType, cls: type[AgentBehaviour]
) -> None:
    agent = _agent(vision_router, agent_type)

    behaviour = behaviour_of(agent)

    assert type(behaviour) is cls
    assert behaviour.agent is agent
    assert behaviour_class(agent_type.value) is cls


def test_every_type_has_a_class() -> None:
    assert {cls.agent_type for cls in map(behaviour_class, AgentType)} == set(AgentType)


def test_the_gui_types_are_the_gui_agent_subtypes(vision_router: Router) -> None:
    assert agent_types_of(GuiAgent) == {
        AgentType.TooledGuiAgent,
        AgentType.TextGuiAgent,
    }
    assert agent_types_of(AgentBehaviour) == set(AgentType)
    assert is_gui_agent(_agent(vision_router, AgentType.TextGuiAgent))
    assert not is_gui_agent(_agent(vision_router, AgentType.Agent))


def test_a_type_runs_as_one_class_only() -> None:
    with pytest.raises(TypeError, match="already runs as DefaultAgent"):

        class _Second(DefaultAgent, agent_type=AgentType.Agent):  # pyright: ignore[reportUnusedClass]
            pass


def test_the_tool_name_comes_from_the_row(vision_router: Router) -> None:
    for agent_type in AgentType:
        agent = _agent(vision_router, agent_type, name="Tooled GuiAgent")
        assert behaviour_of(agent).tool_name == "tooled_guiagent"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_a_plain_agent_keeps_its_fields_and_ignores_none(vision_router: Router) -> None:
    agent = _agent(
        vision_router, AgentType.Agent, arguments=[TASK], output_model=REPORT
    )

    assert validate_agent(agent) == []
    assert dump_fields(agent.arguments) == [TASK]
    assert agent.input_type == Agent.InputType.TEXT


@pytest.mark.parametrize("agent_type", sorted(agent_types_of(GuiAgent)))
def test_a_gui_agent_gets_the_gui_inputs_and_lists_what_it_ignores(
    vision_router: Router, agent_type: AgentType
) -> None:
    agent = _agent(vision_router, agent_type, arguments=[TASK], output_model=REPORT)

    ignored = validate_agent(agent)

    assert dump_fields(agent.arguments) == dump_fields(GUI_ARGUMENTS)
    assert agent.input_type == Agent.InputType.IMAGETEXT
    assert [field.field for field in ignored] == [
        "arguments",
        "input_type",
        "output_model",
        "tools",
        "sub_agents",
    ]
    assert all(field.reason for field in ignored)


@pytest.mark.parametrize("agent_type", sorted(agent_types_of(GuiAgent)))
def test_a_gui_agent_needs_a_vision_router(
    make_router: Callable[..., Router], agent_type: AgentType
) -> None:
    router = make_router(capabilities=[Router.Capability.TEXT], persist=False)

    with pytest.raises(ValueError, match="VISION"):
        _ = validate_agent(_agent(router, agent_type))


def test_a_text_gui_agent_needs_a_parser(vision_router: Router) -> None:
    agent = _agent(vision_router, AgentType.TextGuiAgent, parser_path=None)

    with pytest.raises(ValueError, match="needs a parser_path"):
        _ = validate_agent(agent)


@pytest.mark.parametrize(
    "agent_type", [t for t in AgentType if t is not AgentType.TextGuiAgent]
)
def test_only_a_text_gui_agent_takes_a_parser(
    vision_router: Router, agent_type: AgentType
) -> None:
    agent = _agent(vision_router, agent_type, parser_path=UITARS_PARSER)

    with pytest.raises(ValueError, match=f"set it to null for {agent_type.value}"):
        _ = validate_agent(agent)


PARSERS = "database.agents.behaviours.gui.parsers"


@pytest.mark.parametrize(
    ("parser_path", "message"),
    [
        ("UITarsParser", "as 'module:attribute'"),
        ("os:getcwd", f"must name a module in {PARSERS}"),
        (f"{PARSERS}.missing:Parser", "the module can't be imported"),
        (f"{PARSERS}.uitars:Missing", "has no attribute 'Missing'"),
        (f"{PARSERS}.uitars:WAIT_SECONDS", "doesn't implement the parser interface"),
        (f"{PARSERS}.uitars:ParsedAction", "can't be built without arguments"),
    ],
)
def test_a_parser_that_cant_be_used_is_rejected(
    vision_router: Router, parser_path: str, message: str
) -> None:
    agent = _agent(vision_router, AgentType.TextGuiAgent, parser_path=parser_path)

    with pytest.raises(ValueError, match=message):
        _ = validate_agent(agent)


# ---------------------------------------------------------------------------
# model_validate builds an empty instance first, so no rule runs in __init__
# ---------------------------------------------------------------------------


def test_model_validate_builds_an_agent_and_the_rules_run_after(
    make_router: Callable[..., Router],
) -> None:
    data = {
        "name": "Clicker",
        "description": "d",
        "prompt": "p",
        "input_type": "TEXT",
        "type": "TextGuiAgent",
        "router_id": uuid.uuid4(),
    }

    agent = Agent.model_validate(data)

    assert agent.type == AgentType.TextGuiAgent
    agent.router = make_router(capabilities=[Router.Capability.VISION], persist=False)
    with pytest.raises(ValueError, match="needs a parser_path"):
        _ = validate_agent(agent)
    # The next agent is built from scratch, unaffected by the one before.
    assert Agent.model_validate(data | {"type": "Agent"}).type == AgentType.Agent


def test_model_validate_builds_a_self_link_the_database_rejects(
    session: Session, mock_agent: Agent
) -> None:
    link = SubAgent.model_validate(
        {"parent_agent_id": mock_agent.id, "child_agent_id": mock_agent.id}
    )
    other = SubAgent(parent_agent_id=mock_agent.id, child_agent_id=uuid.uuid4())

    assert (link.parent_agent_id, other.limit) == (mock_agent.id, None)
    session.add(link)
    with pytest.raises(IntegrityError, match="subagent_not_self"):
        session.commit()
    session.rollback()
