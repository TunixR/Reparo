"""Populator for the default recovery agent graph.

Idempotent: nothing is created while a default recovery graph, or a graph with its
name, exists. Agents must have been populated first (populate_agents).
"""

from __future__ import annotations

from sqlalchemy import Engine
from sqlmodel import Session, col, or_, select

from database.agent_graphs.models import AgentGraph, AgentGraphNode
from database.agents.models import Agent, AgentType

DEFAULT_RECOVERY_GRAPH = "Default Recovery"
DEFAULT_RECOVERY_DESCRIPTION = "Runs the gateway agent on every recovery."
GATEWAY_NODE_KEY = "gateway"


def populate_agent_graphs(engine: Engine) -> None:
    """Create the enabled default recovery graph: the gateway agent as its one node."""
    with Session(engine) as session:
        existing = session.exec(
            select(AgentGraph.id).where(
                or_(
                    col(AgentGraph.is_default_recovery).is_(True),
                    AgentGraph.name == DEFAULT_RECOVERY_GRAPH,
                )
            )
        ).first()
        if existing is not None:
            return
        gateway = session.exec(
            select(Agent)
            .where(Agent.type == AgentType.GatewayAgent)
            .order_by(col(Agent.created_at))
        ).first()
        if gateway is None:
            print(
                "[populate_agent_graphs] No gateway agent; no default recovery graph."
            )
            return
        graph = AgentGraph(
            name=DEFAULT_RECOVERY_GRAPH,
            description=DEFAULT_RECOVERY_DESCRIPTION,
            enabled=True,
            is_default_recovery=True,
        )
        graph.nodes = [
            AgentGraphNode(
                agent_graph_id=graph.id,
                key=GATEWAY_NODE_KEY,
                agent_id=gateway.id,
                is_entry=True,
            )
        ]
        session.add(graph)
        session.commit()
