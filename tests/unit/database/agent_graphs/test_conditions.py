"""An edge's condition is a declarative predicate over its source node's output."""

import logging
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import StatementError
from sqlmodel import Session, insert

from database.agent_graphs.conditions import AllOf, AnyOf, Compare, Not
from database.agent_graphs.models import AgentGraph, AgentGraphEdge
from database.agents.models import Agent
from tests.unit.fixtures.agent_graph_fixtures import Edge, MakeAgentGraph, Node
from tests.unit.shared.log_events import logged_events


class Detail(BaseModel):
    code: int
    tags: dict[str, Any]


class Verdict(BaseModel):
    ui_exception: bool
    severity: int
    category: str
    note: str | None
    detail: Detail


OUTPUT = Verdict(
    ui_exception=True,
    severity=3,
    category="ui",
    note=None,
    detail=Detail(code=404, tags={"screen": "login", "retry": None}),
)


def _edge(condition: Any) -> AgentGraphEdge:
    return AgentGraphEdge.model_validate(
        {
            "agent_graph_id": uuid.uuid4(),
            "source_node_id": uuid.uuid4(),
            "target_node_id": uuid.uuid4(),
            "condition": condition,
        }
    )


def _holds(condition: dict[str, Any] | None, output: BaseModel | None = OUTPUT) -> bool:
    return _edge(condition).holds(output)


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("eq", 3, True),
        ("eq", 4, False),
        ("ne", 4, True),
        ("ne", 3, False),
        ("in", [1, 3], True),
        ("in", [1, 2], False),
        ("not_in", [1, 2], True),
        ("not_in", [1, 3], False),
        ("lt", 4, True),
        ("lt", 3, False),
        ("le", 3, True),
        ("le", 2, False),
        ("gt", 2, True),
        ("gt", 3, False),
        ("ge", 3, True),
        ("ge", 4, False),
    ],
)
def test_a_comparison_holds_by_its_operator(op: str, value: Any, expected: bool):
    assert _holds({"field": "severity", "op": op, "value": value}) is expected


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("eq", 1, False),
        ("ne", 1, True),
        ("in", [1], False),
        ("not_in", [1], True),
        ("eq", True, True),
    ],
)
def test_a_boolean_only_equals_a_boolean(op: str, value: Any, expected: bool):
    assert _holds({"field": "ui_exception", "op": op, "value": value}) is expected


@pytest.mark.parametrize(
    ("field", "op", "expected"),
    [
        ("note", "is_null", True),
        ("note", "not_null", False),
        ("category", "is_null", False),
        ("category", "not_null", True),
    ],
)
def test_a_null_check_holds_by_its_operator(field: str, op: str, expected: bool):
    assert _holds({"field": field, "op": op}) is expected


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"field": "detail.code", "op": "eq", "value": 404}, True),
        ({"field": "detail.tags.screen", "op": "in", "value": ["login"]}, True),
        ({"field": "detail.tags.retry", "op": "is_null"}, True),
        ({"field": "detail.tags.screen", "op": "ne", "value": "login"}, False),
    ],
)
def test_a_dotted_field_reaches_into_nested_output(
    condition: dict[str, Any], expected: bool
):
    assert _holds(condition) is expected


TRUE = {"field": "ui_exception", "op": "eq", "value": True}
FALSE = {"field": "category", "op": "eq", "value": "network"}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ({"all": [TRUE, TRUE]}, True),
        ({"all": [TRUE, FALSE]}, False),
        ({"all": []}, True),
        ({"any": [FALSE, TRUE]}, True),
        ({"any": [FALSE, FALSE]}, False),
        ({"any": []}, False),
        ({"not": TRUE}, False),
        ({"not": FALSE}, True),
        ({"all": [TRUE, {"not": {"any": [FALSE, {"not": TRUE}]}}]}, True),
    ],
)
def test_combinators_nest_conditions(condition: dict[str, Any], expected: bool):
    assert _holds(condition) is expected


def _condition_errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return logged_events(caplog, "agent_graph.condition_error")


@pytest.mark.parametrize(
    ("condition", "error"),
    [
        ({"field": "missing", "op": "eq", "value": 1}, "KeyError"),
        ({"field": "missing", "op": "is_null"}, "KeyError"),
        ({"field": "detail.missing", "op": "not_null"}, "KeyError"),
        ({"field": "category.name", "op": "eq", "value": "ui"}, "TypeError"),
        ({"field": "category", "op": "lt", "value": 3}, "TypeError"),
        ({"not": {"field": "missing", "op": "eq", "value": 1}}, "KeyError"),
    ],
)
def test_a_condition_that_raises_is_false_and_logged(
    condition: dict[str, Any], error: str, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    edge = _edge(condition)

    assert edge.holds(OUTPUT) is False

    [record] = _condition_errors(caplog)
    assert record.levelno == logging.WARNING
    assert vars(record)["error.type"] == error
    assert vars(record)["agent_graph.edge.id"] == str(edge.id)


def test_a_condition_over_a_missing_output_is_false(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING)

    assert _holds(TRUE, output=None) is False
    assert _condition_errors(caplog) == []


def test_an_unconditional_edge_always_holds():
    assert _holds(None) is True
    assert _holds(None, output=None) is True


INVALID = [
    {"field": "x", "op": "matches", "value": 1},
    {"field": "x", "op": "in", "value": 3},
    {"field": "x", "op": "not_in", "value": "abc"},
    {"field": "x", "op": "eq", "value": 1, "extra": True},
    {"op": "eq", "value": 1},
    {"all": TRUE},
    {"any": [TRUE, {"field": "x", "op": "matches"}]},
    {"not": [TRUE]},
    {},
    ["not", "a", "condition"],
]


def _bare_edge(condition: Any) -> AgentGraphEdge:
    return AgentGraphEdge(
        agent_graph_id=uuid.uuid4(),
        source_node_id=uuid.uuid4(),
        target_node_id=uuid.uuid4(),
        condition=condition,
    )


@pytest.mark.parametrize("build", [_edge, _bare_edge])
@pytest.mark.parametrize("condition", INVALID)
def test_an_edge_rejects_a_malformed_condition(
    condition: Any, build: Callable[[Any], AgentGraphEdge]
):
    with pytest.raises(ValidationError):
        _ = build(condition)


def test_a_bare_edge_still_evaluates_its_condition():
    assert _bare_edge({"not": FALSE}).holds(OUTPUT) is True


def test_an_edge_validated_from_json_holds_a_typed_predicate():
    edge = AgentGraphEdge.model_validate(
        {
            "agent_graph_id": uuid.uuid4(),
            "source_node_id": uuid.uuid4(),
            "target_node_id": uuid.uuid4(),
            "condition": {"all": [TRUE, {"not": FALSE}]},
        }
    )

    assert isinstance(edge.condition, AllOf)
    assert edge.condition.all == [
        Compare(field="ui_exception", op="eq", value=True),
        Not.model_validate(
            {"not": Compare(field="category", op="eq", value="network")}
        ),
    ]


def test_a_stored_condition_loads_back_as_a_typed_predicate(
    session: Session,
    make_agent: Any,
    mock_router: Any,
    make_agent_graph: MakeAgentGraph,
):
    agent: Agent = make_agent(router=mock_router, name="Only")
    made = make_agent_graph(
        Node("a", agent),
        Node("b", agent),
        edges=[Edge("a", "b", {"any": [TRUE, {"not": FALSE}]})],
    )
    session.expunge_all()

    graph = session.get(AgentGraph, made.id)

    assert graph is not None
    [edge] = graph.entry.outgoing
    assert isinstance(edge.condition, AnyOf)
    assert edge.condition.any == [
        Compare(field="ui_exception", op="eq", value=True),
        Not.model_validate(
            {"not": Compare(field="category", op="eq", value="network")}
        ),
    ]


def test_a_malformed_condition_is_never_stored(
    session: Session,
    make_agent: Any,
    mock_router: Any,
    make_agent_graph: MakeAgentGraph,
):
    agent: Agent = make_agent(router=mock_router, name="Only")
    graph = make_agent_graph(Node("a", agent), Node("b", agent), edges=[])
    a, b = sorted(graph.nodes, key=lambda node: node.key)

    with pytest.raises(StatementError):
        _ = session.exec(
            insert(AgentGraphEdge).values(
                id=uuid.uuid4(),
                agent_graph_id=graph.id,
                source_node_id=a.id,
                target_node_id=b.id,
                priority=0,
                condition={"field": "x", "op": "matches"},
            )
        )
