"""The agent graph populator: the default recovery graph of the gateway."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import Engine
from sqlmodel import Session, select

from database.agent_graphs.models import AgentGraph
from database.agents.models import Agent, AgentType
from database.populators.agent_graphs import populate_agent_graphs
from database.provider.models import Router
from tests.unit.fixtures.agent_graph_fixtures import MakeAgentGraph, Node


def _graphs(session: Session) -> list[AgentGraph]:
    session.expire_all()
    return list(session.exec(select(AgentGraph)))


def test_the_gateway_becomes_the_default_recovery_graph(
    engine: Engine,
    session: Session,
    make_agent: Callable[..., Agent],
    mock_router: Router,
):
    gateway = make_agent(router=mock_router, type=AgentType.GatewayAgent)

    populate_agent_graphs(engine)
    populate_agent_graphs(engine)

    [graph] = _graphs(session)
    assert graph.enabled and graph.is_default_recovery
    assert AgentGraph.default_recovery_id(session) == graph.id
    assert [(n.key, n.agent_id, n.is_entry) for n in graph.nodes] == [
        ("gateway", gateway.id, True)
    ]
    assert graph.edges == []


def test_without_a_gateway_no_graph_is_created(
    engine: Engine,
    session: Session,
    make_agent: Callable[..., Agent],
    mock_router: Router,
):
    _ = make_agent(router=mock_router)

    populate_agent_graphs(engine)

    assert _graphs(session) == []


def test_an_existing_default_recovery_graph_is_kept(
    engine: Engine,
    session: Session,
    make_agent: Callable[..., Agent],
    mock_router: Router,
    make_agent_graph: MakeAgentGraph,
):
    gateway = make_agent(router=mock_router, type=AgentType.GatewayAgent)
    existing = make_agent_graph(Node("entry", gateway), name="Custom Recovery")

    populate_agent_graphs(engine)

    assert [graph.id for graph in _graphs(session)] == [existing.id]
