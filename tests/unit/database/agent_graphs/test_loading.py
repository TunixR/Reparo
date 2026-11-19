"""An agent graph loads everything a run walks in one go."""

from collections.abc import Callable

from sqlmodel import Session

from database.agent_graphs.models import AgentGraph
from database.agents.models import Agent
from database.provider.models import Router
from tests.unit.fixtures.agent_graph_fixtures import MakeAgentGraph, Node


def test_a_loaded_graph_walks_without_its_session(
    session: Session,
    make_agent: Callable[..., Agent],
    mock_router: Router,
    make_agent_graph: MakeAgentGraph,
):
    first = make_agent(router=mock_router, name="First")
    second = make_agent(router=mock_router, name="Second")
    made = make_agent_graph(Node("a", first), Node("b", second), Node("c", first))
    session.expunge_all()

    graph = session.get(AgentGraph, made.id)
    assert graph is not None
    session.expunge_all()

    walked: list[tuple[str, str, Router]] = []
    node = graph.entry
    while node is not None:
        walked.append((node.key, node.agent.name, node.agent.router))
        node = node.next_node()
    assert [(key, name) for key, name, _ in walked] == [
        ("a", "First"),
        ("b", "Second"),
        ("c", "First"),
    ]
