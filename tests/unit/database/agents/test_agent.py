from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel
from sqlmodel import Session
from strands.types.tools import JSONSchema

from database.agents.behaviours import behaviour_of, validate_agent
from database.agents.behaviours.gui.shared import GuiRunReport
from database.agents.models import Agent, AgentType, AgentUpdate
from database.agents.specs import OutputModelSpec
from database.logging.models import (
    ToolTrace,  # pyright: ignore[reportUnusedImport] # noqa: F401 Needs to be imported to register with SQLModel
)
from database.provider.models import Router


def _make_router() -> Router:
    return Router(
        api_key="test-key",
        model_name="test-model",
        api_endpoint="https://api.test/v1",
        provider_type=Router.Provider.OPENAI,
    )


ARGUMENTS: list[dict[str, Any]] = [
    {"name": "task", "type": "str", "description": "Task description"},
    {"name": "attempts", "type": "int", "description": "Attempts so far"},
    {"name": "budget", "type": "float", "description": "Budget left"},
    {"name": "history", "type": "list[str]", "description": "Prior actions"},
    {"name": "mode", "type": "Literal['fast', 'safe']", "description": "Mode"},
    {"name": "note", "type": "str | None", "description": "Optional note"},
    {"name": "act", "type": "dict[str, Any]", "description": "The failed act"},
    {"name": "retry", "type": "bool", "description": "Retry", "default": False},
]


def _make_agent(router: Router, **overrides: Any) -> Agent:
    """An in-memory agent (not persisted) for validation tests."""
    return Agent(
        name="Validation Test Agent",
        description="Agent used for testing validation logic.",
        prompt="You are a test agent.",
        enabled=True,
        router=router,
        **(
            {
                "type": AgentType.Agent,
                "input_type": Agent.InputType.TEXT,
                "arguments": ARGUMENTS,
            }
            | overrides
        ),
    )


def valid_kwargs() -> dict[str, Any]:
    return {
        "task": "Process invoice",
        "attempts": 2,
        "budget": 1.5,
        "history": ["open_app", "login"],
        "mode": "safe",
        "note": None,
        "act": {"name": "click_submit", "error": "ElementNotFound"},
    }


class _ErroredAct(BaseModel):
    name: str
    at: datetime


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_validate_input_returns_the_validated_inputs_with_defaults() -> None:
    agent = _make_agent(_make_router())

    assert agent.validate_input(**valid_kwargs()) == valid_kwargs() | {"retry": False}


def test_validate_input_names_a_missing_argument() -> None:
    agent = _make_agent(_make_router())
    inputs = valid_kwargs()
    del inputs["act"]

    with pytest.raises(ValueError, match="Missing required argument: act"):
        _ = agent.validate_input(**inputs)


def test_none_is_a_value_when_the_type_allows_it() -> None:
    agent = _make_agent(_make_router())

    _ = agent.validate_input(**valid_kwargs() | {"note": None})
    with pytest.raises(ValueError, match="task"):
        _ = agent.validate_input(**valid_kwargs() | {"task": None})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("attempts", True),
        ("attempts", "2"),
        ("budget", "1.5"),
        ("history", "not-a-list"),
        ("history", [1]),
        ("mode", "typo"),
        ("act", ["not", "a", "dict"]),
    ],
)
def test_validate_input_is_strict(name: str, value: Any) -> None:
    agent = _make_agent(_make_router())

    with pytest.raises(ValueError, match=f"Argument '{name}'"):
        _ = agent.validate_input(**valid_kwargs() | {name: value})


def test_an_int_passes_a_float_argument() -> None:
    agent = _make_agent(_make_router())

    assert agent.validate_input(**valid_kwargs() | {"budget": 3})["budget"] == 3


def test_validate_input_rejects_an_unknown_argument() -> None:
    agent = _make_agent(_make_router())

    with pytest.raises(ValueError, match="Unexpected argument: extra"):
        _ = agent.validate_input(**valid_kwargs(), extra=1)


def test_a_model_from_the_graph_context_is_validated_as_its_json() -> None:
    agent = _make_agent(_make_router())
    act = _ErroredAct(name="click", at=datetime(2026, 1, 2, tzinfo=UTC))

    inputs = agent.validate_input(**valid_kwargs() | {"act": act})

    assert inputs["act"] == {"name": "click", "at": "2026-01-02T00:00:00Z"}


# ---------------------------------------------------------------------------
# Argument and output model definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argument", "message"),
    [
        ({"type": "bytes"}, "unknown type 'bytes'"),
        ({"type": "__import__('os')"}, "__import__"),
        ({"type": "int", "default": "3"}, "default '3' is not a int"),
        ({"name": "has space"}, "has space"),
        ({"json_type": "string"}, "json_type"),
    ],
)
def test_construction_rejects_a_bad_argument_naming_the_field(
    argument: dict[str, Any], message: str
) -> None:
    bad = {"name": "task", "type": "str", "description": "Task"} | argument

    with pytest.raises(ValueError, match="arguments") as info:
        _ = _make_agent(_make_router(), arguments=[bad])
    assert message in str(info.value)


def test_construction_rejects_a_duplicate_argument() -> None:
    task = {"name": "task", "type": "str", "description": "Task"}

    with pytest.raises(ValueError, match="'task' is defined more than once"):
        _ = _make_agent(_make_router(), arguments=[task, task])


def test_construction_rejects_a_non_identifier_output_model_name() -> None:
    with pytest.raises(ValueError, match="output_model") as info:
        _ = _make_agent(
            _make_router(), output_model={"name": "Triage Report", "fields": []}
        )
    assert "'Triage Report' is not a Python identifier" in str(info.value)


def test_assignment_validates_like_construction() -> None:
    agent = _make_agent(_make_router())

    with pytest.raises(ValueError, match="unknown type 'set'|set"):
        agent.arguments = [{"name": "x", "type": "set[str]", "description": "X"}]  # pyright: ignore[reportAttributeAccessIssue] - the JSON form converts


def test_definitions_persist_in_their_normalized_form(
    session: Session, make_agent: Callable[..., Agent], mock_router: Router
) -> None:
    agent = make_agent(
        router=mock_router,
        arguments=[{"name": "note", "type": "str|None", "description": "Note"}],
        output_model=REPORT,
    )
    session.expire_all()

    persisted = session.get(Agent, agent.id)

    assert persisted is not None
    assert [spec.model_dump() for spec in persisted.arguments] == [
        {"name": "note", "type": "str | None", "description": "Note"}
    ]
    assert persisted.output_model == OutputModelSpec.model_validate(REPORT)


REPORT: dict[str, Any] = {
    "name": "TriageReport",
    "description": "How the failure was triaged.",
    "fields": [
        {"name": "kind", "type": "Literal['ui', 'data']", "description": "Kind"},
        {"name": "note", "type": "str | None", "description": "Note"},
    ],
}


def test_the_output_class_is_built_from_the_output_model() -> None:
    agent = _make_agent(_make_router(), output_model=REPORT)

    report = behaviour_of(agent).output_class()

    assert report is not None
    assert report.__name__ == "TriageReport"
    assert report.model_validate({"kind": "ui", "note": None}).model_dump() == {
        "kind": "ui",
        "note": None,
    }
    assert behaviour_of(_make_agent(_make_router())).output_class() is None


def test_a_gui_agent_always_reports_a_gui_run_report() -> None:
    agent = _make_agent(
        _make_router(), type=AgentType.TooledGuiAgent, output_model=REPORT
    )

    assert behaviour_of(agent).output_class() is GuiRunReport


# ---------------------------------------------------------------------------
# Schema generation tests
# ---------------------------------------------------------------------------


def test_get_input_schema_describes_the_arguments() -> None:
    agent = _make_agent(_make_router())

    schema: dict[str, Any] = agent.get_input_schema()["json"]

    assert schema["type"] == "object"
    assert list(schema["properties"]) == [a["name"] for a in ARGUMENTS]
    assert schema["required"] == [
        "task",
        "attempts",
        "budget",
        "history",
        "mode",
        "note",
        "act",
    ]
    props = schema["properties"]
    assert props["history"]["items"] == {"type": "string"}
    assert props["mode"]["enum"] == ["fast", "safe"]
    assert props["task"]["description"] == "Task description"
    assert props["retry"]["default"] is False


def test_as_tool_reflects_agent_metadata():
    agent = _make_agent(_make_router())
    decorated = behaviour_of(agent).as_tool()
    assert decorated._tool_name == "validation_test_agent"  # pyright: ignore[reportPrivateUsage]
    assert decorated._tool_spec.get("description", "") == agent.description  # pyright: ignore[reportPrivateUsage]
    # Input schema from decoration should mirror dynamic schema
    tool_schema = decorated._tool_spec.get("inputSchema", JSONSchema()).get("json", {})  # pyright: ignore[reportPrivateUsage]
    agent_schema = agent.get_input_schema().get("json", {})
    assert tool_schema == agent_schema


# ---------------------------------------------------------------------------
# Router capability alignment
# ---------------------------------------------------------------------------


def test_text_agent_aligns_with_text_capability(
    make_agent: Callable[..., Agent],
    make_router: Callable[..., Router],
):
    agent = make_agent(
        router=make_router(),
        input_type=Agent.InputType.TEXT,
    )

    assert agent.input_type == Agent.InputType.TEXT


def test_validate_agent_rejects_a_text_agent_on_a_vision_only_router(
    make_router: Callable[..., Router],
):
    router = make_router(capabilities=[Router.Capability.VISION], persist=False)
    agent = _make_agent(router, input_type=Agent.InputType.TEXT)

    with pytest.raises(ValueError, match="TEXT"):
        _ = validate_agent(agent)


def test_validate_agent_rejects_an_image_agent_without_vision(
    make_router: Callable[..., Router],
):
    router = make_router(capabilities=[Router.Capability.TEXT], persist=False)
    agent = _make_agent(router, input_type=Agent.InputType.IMAGETEXT)

    with pytest.raises(ValueError, match="VISION"):
        _ = validate_agent(agent)


def test_imagtext_agent_aligns_with_text_and_vision_capabilities(
    make_agent: Callable[..., Agent],
    make_router: Callable[..., Router],
):
    agent = make_agent(
        router=make_router(
            capabilities=[Router.Capability.TEXT, Router.Capability.VISION]
        ),
        input_type=Agent.InputType.IMAGETEXT,
    )

    assert agent.input_type == Agent.InputType.IMAGETEXT


def test_an_update_to_an_input_type_the_router_lacks_fails_validation(
    make_agent: Callable[..., Agent],
    make_router: Callable[..., Router],
):
    router = make_router(capabilities=[Router.Capability.TEXT])
    agent = make_agent(router=router, input_type=Agent.InputType.TEXT)

    agent.apply_update(AgentUpdate(input_type=Agent.InputType.IMAGETEXT), router)

    with pytest.raises(ValueError, match="VISION"):
        _ = validate_agent(agent)


def test_agent_update_accepts_compatible_input_type(
    session: Session,
    make_agent: Callable[..., Agent],
    make_router: Callable[..., Router],
):
    router = make_router(
        capabilities=[Router.Capability.TEXT, Router.Capability.VISION]
    )
    agent = make_agent(router=router, input_type=Agent.InputType.TEXT)

    agent.apply_update(
        AgentUpdate(input_type=Agent.InputType.IMAGETEXT),
        router,
    )
    session.add(agent)
    session.commit()

    assert agent.input_type == Agent.InputType.IMAGETEXT


def test_agent_reasoning_effort_defaults_to_provider(
    make_agent: Callable[..., Agent], mock_router: Router
):
    agent = make_agent(router=mock_router)

    assert agent.reasoning_effort is None
    assert agent.get_model_params() == {}
    assert "reasoning_effort" not in agent.get_model().get_config().get("params", {})


def test_agent_reasoning_effort_persists_and_reaches_model_params(
    session: Session, make_agent: Callable[..., Agent], mock_router: Router
):
    agent = make_agent(
        router=mock_router, reasoning_effort=Agent.ReasoningEffort.MEDIUM
    )
    session.expire_all()

    persisted = session.get(Agent, agent.id)

    assert persisted is not None
    assert persisted.get_model().get_config()["params"] == {
        "reasoning_effort": "medium"
    }


def test_agent_apply_update_clears_reasoning_effort_on_explicit_null():
    router = _make_router()
    agent = Agent(
        name="Agent",
        description="d",
        prompt="p",
        input_type=Agent.InputType.TEXT,
        type=AgentType.Agent,
        router=router,
        reasoning_effort=Agent.ReasoningEffort.HIGH,
    )

    agent.apply_update(AgentUpdate(name="Renamed"), router)
    assert agent.reasoning_effort == Agent.ReasoningEffort.HIGH

    agent.apply_update(AgentUpdate(reasoning_effort=None), router)
    assert agent.reasoning_effort is None


def test_apply_update_replaces_the_definitions_and_clears_the_output_model() -> None:
    router = _make_router()
    agent = _make_agent(router, output_model=REPORT)
    task = {"name": "task", "type": "str", "description": "Task"}

    agent.apply_update(AgentUpdate.model_validate({"arguments": [task]}), router)
    assert [spec.name for spec in agent.arguments] == ["task"]
    assert agent.output_model is not None

    agent.apply_update(AgentUpdate.model_validate({"output_model": None}), router)
    assert agent.output_model is None


def test_apply_update_keeps_a_null_default() -> None:
    router = _make_router()
    agent = _make_agent(router)
    note = {
        "name": "note",
        "type": "str | None",
        "description": "Note",
        "default": None,
    }

    agent.apply_update(AgentUpdate.model_validate({"arguments": [note]}), router)

    assert agent.arguments[0].has_default
