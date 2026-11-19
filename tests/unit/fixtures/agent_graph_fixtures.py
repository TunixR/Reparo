"""
Agent graph fixtures for unit tests.
"""

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlmodel import Session, col, select

from database.agent_graphs.models import AgentGraph, AgentGraphEdge, AgentGraphNode
from database.agents.models import Agent


@dataclass(frozen=True)
class Node:
    """A node to create: its key, its agent and its input map."""

    key: str
    agent: Agent
    input_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """An edge to create; without a priority, its place in ``edges`` is its priority."""

    source: str
    target: str
    condition: dict[str, Any] | None = None
    priority: int | None = None


type MakeAgentGraph = Callable[..., AgentGraph]
"""Factory returned by ``make_agent_graph``: nodes and edges in, the persisted graph out."""


@pytest.fixture
def make_agent_graph(session: Session) -> MakeAgentGraph:
    """Return a factory that persists an agent graph.

    The first node is the entry node. ``edges`` are `Edge`s or unconditional
    ``(source, target)`` key pairs; without them, the nodes are chained in order. A
    new default recovery graph takes over from the previous one.
    """

    def _make_agent_graph(
        *nodes: Node,
        edges: Sequence[Edge | tuple[str, str]] | None = None,
        name: str | None = None,
        enabled: bool = True,
        is_default_recovery: bool = True,
        max_node_executions: int | None = None,
        execution_timeout_s: float | None = None,
        node_timeout_s: float | None = None,
    ) -> AgentGraph:
        if is_default_recovery:
            for previous in session.exec(
                select(AgentGraph).where(col(AgentGraph.is_default_recovery).is_(True))
            ):
                previous.is_default_recovery = False
                session.add(previous)
            session.commit()
        graph = AgentGraph(
            name=name or f"Test Graph {uuid.uuid4()}",
            description="A graph for testing.",
            enabled=enabled,
            is_default_recovery=is_default_recovery,
            max_node_executions=max_node_executions,
            execution_timeout_s=execution_timeout_s,
            node_timeout_s=node_timeout_s,
        )
        rows = {
            node.key: AgentGraphNode(
                agent_graph_id=graph.id,
                key=node.key,
                agent_id=node.agent.id,
                is_entry=index == 0,
                input_map=dict(node.input_map),
            )
            for index, node in enumerate(nodes)
        }
        graph.nodes = list(rows.values())
        if edges is None:
            edges = [(a.key, b.key) for a, b in zip(nodes, nodes[1:], strict=False)]
        normalized = [edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges]
        graph.edges = [
            AgentGraphEdge(
                agent_graph_id=graph.id,
                source_node_id=rows[edge.source].id,
                target_node_id=rows[edge.target].id,
                priority=index if edge.priority is None else edge.priority,
                condition=edge.condition,
            )
            for index, edge in enumerate(normalized)
        ]
        session.add(graph)
        session.commit()
        session.refresh(graph)
        return graph

    return _make_agent_graph
