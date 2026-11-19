"""Load the agent graph a recovery runs, with every row the run needs."""

from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from database.agent_graphs.models import AgentGraph, AgentGraphNode
from database.agent_graphs.runner import GraphEdge, GraphNode, LoadedAgentGraph
from database.agents.models import Agent

FALLBACK_NODE_KEY = "gateway"


def load_agent_graph(session: Session, gateway: Agent) -> LoadedAgentGraph:
    """The enabled agent graph entered at ``gateway``, else a one-node graph of it.

    Nodes, edges and node agents (with their routers, tools and arguments) are
    loaded here, so nothing lazy-loads during the run. The node cap is the graph's
    ``max_node_executions``, or its node count.
    """
    graph = session.exec(
        select(AgentGraph)
        .join(AgentGraphNode)
        .where(
            col(AgentGraph.enabled).is_(True),
            col(AgentGraphNode.is_entry).is_(True),
            AgentGraphNode.agent_id == gateway.id,
        )
        .options(
            selectinload(AgentGraph.nodes).joinedload(AgentGraphNode.agent),  # pyright: ignore[reportArgumentType]
            selectinload(AgentGraph.edges),  # pyright: ignore[reportArgumentType]
        )
    ).first()
    if graph is None:
        return LoadedAgentGraph.single(FALLBACK_NODE_KEY, gateway)

    keys = {node.id: node.key for node in graph.nodes}
    return LoadedAgentGraph(
        nodes=tuple(
            GraphNode(key=node.key, agent=node.agent, input_map=dict(node.input_map))
            for node in graph.nodes
        ),
        edges=tuple(
            GraphEdge(
                source=keys[edge.source_node_id],
                target=keys[edge.target_node_id],
                priority=edge.priority,
                condition=edge.condition,
            )
            for edge in graph.edges
        ),
        max_node_executions=graph.max_node_executions or len(graph.nodes),
        execution_timeout_s=graph.execution_timeout_s,
        node_timeout_s=graph.node_timeout_s,
    )
