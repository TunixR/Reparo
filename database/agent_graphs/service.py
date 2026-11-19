"""The rules that keep agent graphs consistent, and the writes that enforce them.

A disabled graph is a draft and accepts any structure. Enabling a graph, and every
write to an enabled one, must leave it valid: its node agents enabled and the
structural rules passing. A rejected write raises `AgentGraphError`
with the violations; the caller rolls back on it and commits otherwise.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastapi import status
from sqlmodel import Session, col, select

from database.agent_graphs.models import (
    AgentGraph,
    AgentGraphCreate,
    AgentGraphEdge,
    AgentGraphEdgeCreate,
    AgentGraphEdgeUpdate,
    AgentGraphNode,
    AgentGraphNodeCreate,
    AgentGraphNodeUpdate,
    AgentGraphUpdate,
)
from database.agents.models import Agent, AgentType

MAX_NODE_EXECUTIONS = 100
"""The most node executions a graph may allow per run (S7)."""


@dataclass(frozen=True, slots=True)
class Violation:
    """A broken rule, by its id in the agent graph spec, and how to fix it."""

    rule: str
    message: str


@dataclass(slots=True)
class AgentGraphError(Exception):
    """A rejected agent graph write, with the HTTP status that fits it."""

    status_code: int
    message: str
    violations: Sequence[Violation] = field(default_factory=tuple)

    def detail(self) -> dict[str, object]:
        return {
            "message": self.message,
            "violations": [
                {"rule": v.rule, "message": v.message} for v in self.violations
            ],
        }


def create_graph(session: Session, payload: AgentGraphCreate) -> AgentGraph:
    data = payload.model_dump(exclude={"is_default_recovery"})
    graph = AgentGraph(**data)
    with session.no_autoflush:
        _ensure_unique_name(session, graph.name, graph.id)
        session.add(graph)
        if payload.is_default_recovery:
            _take_default(session, graph)
        _validate_write(session, graph, enabling=graph.enabled)
    return graph


def update_graph(
    session: Session, graph: AgentGraph, payload: AgentGraphUpdate
) -> AgentGraph:
    data = payload.model_dump(exclude_unset=True)
    data = _drop_nulls(
        data, keep={"max_node_executions", "execution_timeout_s", "node_timeout_s"}
    )
    make_default = data.pop("is_default_recovery", None)
    enabling = not graph.enabled and data.get("enabled") is True
    with session.no_autoflush:
        if payload.name is not None:
            _ensure_unique_name(session, payload.name, graph.id)
        for key, value in data.items():
            setattr(graph, key, value)
        if make_default is False:
            graph.is_default_recovery = False
        elif make_default:
            _take_default(session, graph)
        _validate_write(session, graph, enabling=enabling)
    return graph


def delete_graph(session: Session, graph: AgentGraph) -> None:
    """Delete the graph; its nodes and edges go with it."""
    session.delete(graph)


def add_node(
    session: Session, graph: AgentGraph, payload: AgentGraphNodeCreate
) -> AgentGraphNode:
    with session.no_autoflush:
        agent = _agent(session, payload.agent_id)
        _ensure_unique_key(graph, payload.key, None)
        node = AgentGraphNode(agent_graph_id=graph.id, **payload.model_dump())
        node.agent = agent
        graph.nodes.append(node)
        _validate_write(session, graph, written=node)
    return node


def update_node(
    session: Session,
    graph: AgentGraph,
    node: AgentGraphNode,
    payload: AgentGraphNodeUpdate,
) -> AgentGraphNode:
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    with session.no_autoflush:
        if "key" in data:
            _ensure_unique_key(graph, data["key"], node.id)
        if "agent_id" in data:
            node.agent = _agent(session, data["agent_id"])
        for key, value in data.items():
            setattr(node, key, value)
        _validate_write(session, graph, written=node)
    return node


def remove_node(session: Session, graph: AgentGraph, node: AgentGraphNode) -> None:
    """Remove the node and every edge into or out of it."""
    with session.no_autoflush:
        for edge in [e for e in graph.edges if node.id in _ends(e)]:
            graph.edges.remove(edge)
            session.delete(edge)
        graph.nodes.remove(node)
        session.delete(node)
        _validate_write(session, graph)


def add_edge(
    session: Session, graph: AgentGraph, payload: AgentGraphEdgeCreate
) -> AgentGraphEdge:
    with session.no_autoflush:
        edge = AgentGraphEdge.model_validate(
            payload.model_dump(by_alias=True) | {"agent_graph_id": graph.id}
        )
        _ensure_edge_fits(graph, edge)
        graph.edges.append(edge)
        _validate_write(session, graph)
    return edge


def update_edge(
    session: Session,
    graph: AgentGraph,
    edge: AgentGraphEdge,
    payload: AgentGraphEdgeUpdate,
) -> AgentGraphEdge:
    data = payload.model_dump(exclude_unset=True, by_alias=True)
    data = _drop_nulls(data, keep={"condition"})
    with session.no_autoflush:
        for key, value in data.items():
            setattr(edge, key, value)
        _ensure_edge_fits(graph, edge)
        _validate_write(session, graph)
    return edge


def remove_edge(session: Session, graph: AgentGraph, edge: AgentGraphEdge) -> None:
    with session.no_autoflush:
        graph.edges.remove(edge)
        session.delete(edge)
        _validate_write(session, graph)


def violations(session: Session, graph: AgentGraph) -> list[Violation]:
    """Why ``graph`` may not be enabled: disabled node agents (V4) and broken
    structure rules."""
    return _agent_violations(graph.nodes, "V4") + _structure_violations(session, graph)


def _structure_violations(session: Session, graph: AgentGraph) -> list[Violation]:
    return [
        *_entry_violations(graph),
        *_reachability_violations(graph),
        *_else_violations(graph),
        *_cycle_violations(graph),
        *_gateway_violations(session, graph),
    ]


def _agent_violations(nodes: Sequence[AgentGraphNode], rule: str) -> list[Violation]:
    return [
        Violation(
            rule,
            f"Node '{node.key}' runs the disabled agent '{node.agent.name}'; enable "
            "the agent or use another one.",
        )
        for node in nodes
        if not node.agent.enabled
    ]


def _take_default(session: Session, graph: AgentGraph) -> None:
    """Make ``graph`` the default recovery graph in place of the previous one."""
    if graph.is_default_recovery:
        return
    for previous in session.exec(
        select(AgentGraph).where(
            col(AgentGraph.is_default_recovery).is_(True), AgentGraph.id != graph.id
        )
    ):
        previous.is_default_recovery = False
    # The partial unique index allows one default at a time, so the previous
    # default is cleared before this graph takes over.
    session.flush()
    graph.is_default_recovery = True


def _validate_write(
    session: Session,
    graph: AgentGraph,
    *,
    enabling: bool = False,
    written: AgentGraphNode | None = None,
) -> None:
    """Reject the write if it leaves an enabled ``graph`` invalid.

    Enabling checks every node agent (V4); a write to an enabled graph checks the
    agent of the ``written`` node (V5). Either way the structure must pass.
    """
    graph.updated_at = datetime.now(UTC)
    if not graph.enabled:
        return
    if enabling:
        found = violations(session, graph)
    else:
        written_nodes = [] if written is None else [written]
        found = _agent_violations(written_nodes, "V5")
        found += _structure_violations(session, graph)
    if found:
        action = "can't be enabled" if enabling else "would no longer be valid"
        raise AgentGraphError(
            status.HTTP_409_CONFLICT,
            f"The agent graph '{graph.name}' {action}; fix the violations, or "
            "disable the graph to edit it as a draft.",
            found,
        )


def _entry_violations(graph: AgentGraph) -> list[Violation]:
    """S1: exactly one entry node, with no incoming edges."""
    entries = [node for node in graph.nodes if node.is_entry]
    if not entries:
        return [
            Violation("S1", "The graph has no entry node; mark one node as the entry.")
        ]
    if len(entries) > 1:
        keys = ", ".join(f"'{node.key}'" for node in entries)
        return [
            Violation(
                "S1",
                f"The graph has {len(entries)} entry nodes ({keys}); keep exactly one.",
            )
        ]
    entry = entries[0]
    incoming = [e for e in graph.edges if e.target_node_id == entry.id]
    if incoming:
        return [
            Violation(
                "S1",
                f"The entry node '{entry.key}' has {len(incoming)} incoming edge(s); "
                "a run only starts there, so remove them.",
            )
        ]
    return []


def _reachability_violations(graph: AgentGraph) -> list[Violation]:
    """S2: every node is reachable from the entry, ignoring conditions."""
    entries = [node for node in graph.nodes if node.is_entry]
    if len(entries) != 1:
        return []
    reached = _reachable(graph, entries[0].id)
    return [
        Violation(
            "S2",
            f"Node '{node.key}' is unreachable from the entry node "
            f"'{entries[0].key}'; add an edge into it or remove it.",
        )
        for node in graph.nodes
        if node.id not in reached
    ]


def _else_violations(graph: AgentGraph) -> list[Violation]:
    """S4: at most one unconditional outgoing edge, evaluated last."""
    found: list[Violation] = []
    for node in graph.nodes:
        outgoing = [e for e in graph.edges if e.source_node_id == node.id]
        unconditional = [e for e in outgoing if e.condition is None]
        if len(unconditional) > 1:
            found.append(
                Violation(
                    "S4",
                    f"Node '{node.key}' has {len(unconditional)} unconditional "
                    "outgoing edges; only the first could ever run, so keep one "
                    "and give the others conditions.",
                )
            )
        elif unconditional and any(
            e.priority > unconditional[0].priority for e in outgoing
        ):
            found.append(
                Violation(
                    "S4",
                    f"The unconditional edge of node '{node.key}' must have the "
                    "lowest priority (the highest number), since edges after it are "
                    "never evaluated.",
                )
            )
    return found


def _cycle_violations(graph: AgentGraph) -> list[Violation]:
    """S7: cycles only with ``max_node_executions``, which has a maximum."""
    found: list[Violation] = []
    limit = graph.max_node_executions
    if limit is not None and limit > MAX_NODE_EXECUTIONS:
        found.append(
            Violation(
                "S7",
                f"max_node_executions is {limit}, above the maximum of "
                f"{MAX_NODE_EXECUTIONS}; lower it.",
            )
        )
    if limit is None and _has_cycle(graph):
        found.append(
            Violation(
                "S7",
                "The graph has a cycle; set max_node_executions to bound its runs.",
            )
        )
    return found


def _gateway_violations(session: Session, graph: AgentGraph) -> list[Violation]:
    """S8: a gateway only as the entry node, and of one enabled graph at most."""
    found: list[Violation] = []
    for node in graph.nodes:
        if node.agent.type != AgentType.GatewayAgent:
            continue
        if not node.is_entry:
            found.append(
                Violation(
                    "S8",
                    f"Node '{node.key}' runs the gateway '{node.agent.name}', which "
                    "may only be the entry node.",
                )
            )
            continue
        others = session.exec(
            select(AgentGraph.name)
            .join(AgentGraphNode)
            .where(
                AgentGraphNode.agent_id == node.agent_id,
                col(AgentGraphNode.is_entry).is_(True),
                col(AgentGraph.enabled).is_(True),
                AgentGraph.id != graph.id,
            )
        ).all()
        if others:
            names = ", ".join(f"'{name}'" for name in others)
            found.append(
                Violation(
                    "S8",
                    f"The gateway '{node.agent.name}' already enters the enabled "
                    f"graph {names}; disable that graph first.",
                )
            )
    return found


def _successors(graph: AgentGraph) -> dict[uuid.UUID, list[uuid.UUID]]:
    successors: dict[uuid.UUID, list[uuid.UUID]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        successors.setdefault(edge.source_node_id, []).append(edge.target_node_id)
    return successors


def _reachable(graph: AgentGraph, start: uuid.UUID) -> set[uuid.UUID]:
    successors = _successors(graph)
    reached = {start}
    pending = [start]
    while pending:
        for target in successors.get(pending.pop(), []):
            if target not in reached:
                reached.add(target)
                pending.append(target)
    return reached


def _has_cycle(graph: AgentGraph) -> bool:
    successors = _successors(graph)
    done: set[uuid.UUID] = set()

    def visit(node: uuid.UUID, path: set[uuid.UUID]) -> bool:
        if node in path:
            return True
        if node in done:
            return False
        path.add(node)
        cyclic = any(visit(target, path) for target in successors.get(node, []))
        path.discard(node)
        done.add(node)
        return cyclic

    return any(visit(node, set()) for node in list(successors))


def _violation_error(status_code: int, rule: str, message: str) -> AgentGraphError:
    return AgentGraphError(status_code, message, (Violation(rule, message),))


def _agent(session: Session, agent_id: uuid.UUID) -> Agent:
    agent = session.get(Agent, agent_id)
    if agent is None:
        raise AgentGraphError(status.HTTP_404_NOT_FOUND, f"Agent {agent_id} not found.")
    return agent


def _ensure_unique_name(session: Session, name: str, graph_id: uuid.UUID) -> None:
    taken = session.exec(
        select(AgentGraph.id).where(AgentGraph.name == name, AgentGraph.id != graph_id)
    ).first()
    if taken is not None:
        raise AgentGraphError(
            status.HTTP_409_CONFLICT,
            f"An agent graph named '{name}' already exists; choose another name.",
        )


def _ensure_unique_key(graph: AgentGraph, key: str, node_id: uuid.UUID | None) -> None:
    if any(n.key == key and n.id != node_id for n in graph.nodes):
        raise AgentGraphError(
            status.HTTP_409_CONFLICT,
            f"The graph already has a node '{key}'; node keys are unique in a graph.",
        )


def _ensure_edge_fits(graph: AgentGraph, edge: AgentGraphEdge) -> None:
    """S3, checked even in drafts since the database enforces it too."""
    nodes = {n.id for n in graph.nodes}
    for end in _ends(edge):
        if end not in nodes:
            raise _violation_error(
                status.HTTP_409_CONFLICT,
                "S3",
                f"Node {end} isn't a node of the graph '{graph.name}'; edges stay "
                "inside one graph.",
            )
    for other in _others(graph, edge):
        if other.source_node_id != edge.source_node_id:
            continue
        if other.target_node_id == edge.target_node_id:
            raise _violation_error(
                status.HTTP_409_CONFLICT,
                "S3",
                "An edge between these nodes already exists; change that edge instead.",
            )
        if other.priority == edge.priority:
            raise _violation_error(
                status.HTTP_409_CONFLICT,
                "S3",
                f"The source node already has an edge with priority {edge.priority}; "
                "give each outgoing edge its own priority.",
            )


def _drop_nulls(data: dict[str, Any], *, keep: set[str]) -> dict[str, Any]:
    """Drop explicit nulls, except for fields where null is a value."""
    return {k: v for k, v in data.items() if v is not None or k in keep}


def _others(graph: AgentGraph, edge: AgentGraphEdge) -> Iterator[AgentGraphEdge]:
    return (e for e in graph.edges if e.id != edge.id)


def _ends(edge: AgentGraphEdge) -> tuple[uuid.UUID, uuid.UUID]:
    return edge.source_node_id, edge.target_node_id
