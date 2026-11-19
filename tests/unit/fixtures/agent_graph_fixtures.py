"""
Agent graph fixtures for unit tests.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pytest
from sqlmodel import Session

from database.agent_graphs.models import AgentGraph, AgentGraphEdge, AgentGraphNode
from database.agents.models import Agent


@dataclass(frozen=True)
class Node:
    """A node to create: its key, its agent and its input map."""

    key: str
    agent: Agent
    input_map: Mapping[str, str] = field(default_factory=dict)


type MakeAgentGraph = Callable[..., AgentGraph]
"""Factory returned by ``make_agent_graph``: nodes and edges in, the persisted graph out."""


@pytest.fixture
def make_agent_graph(session: Session) -> MakeAgentGraph:
    """Return a factory that persists an agent graph.

    The first node is the entry node. ``edges`` are unconditional ``(source,
    target)`` key pairs; without them, the nodes are chained in order.
    """

    def _make_agent_graph(
        *nodes: Node,
        edges: Sequence[tuple[str, str]] | None = None,
        name: str = "Test Graph",
        enabled: bool = True,
        max_node_executions: int | None = None,
        execution_timeout_s: float | None = None,
        node_timeout_s: float | None = None,
    ) -> AgentGraph:
        graph = AgentGraph(
            name=name,
            description="A graph for testing.",
            enabled=enabled,
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
        graph.edges = [
            AgentGraphEdge(
                agent_graph_id=graph.id,
                source_node_id=rows[source].id,
                target_node_id=rows[target].id,
                priority=index,
            )
            for index, (source, target) in enumerate(edges)
        ]
        session.add(graph)
        session.commit()
        session.refresh(graph)
        return graph

    return _make_agent_graph
