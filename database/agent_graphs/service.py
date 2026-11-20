"""The rules that keep agent graphs consistent, and the writes that enforce them.

A disabled graph is a draft and accepts any structure. Enabling a graph, and every
write to an enabled one, must leave it valid: its node agents enabled and the
structural rules passing. Writes to an agent that graphs use must not break them
either. A rejected write raises `AgentGraphError` with the violations; the caller
rolls back on it and commits otherwise.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import status
from pydantic import BaseModel
from sqlmodel import Session, col, select

from database.agent_graphs.conditions import (
    CONDITION,
    AllOf,
    AnyOf,
    Compare,
    Condition,
    Not,
)
from database.agent_graphs.field_types import (
    FieldPathError,
    comparison_error,
    field_type,
)
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
from database.agent_graphs.runner import RECOVERY_INPUTS_SOURCE
from database.agents.models import Agent, AgentUpdate
from database.provider.models import Router
from templates.recovery_payload import RecoveryInputs

MAX_NODE_EXECUTIONS = 100
"""The most node executions a graph may allow per run."""


@dataclass(slots=True)
class AgentGraphError(Exception):
    """A rejected write to an agent graph, or to an agent that graphs use: why, each
    broken rule with its fix, and the graphs involved."""

    status_code: int
    message: str
    violations: Sequence[str] = field(default_factory=tuple)
    graphs: Sequence[AgentGraph] = field(default_factory=tuple)

    def detail(self) -> dict[str, object]:
        return {
            "message": self.message,
            "violations": list(self.violations),
            "graphs": [
                {"id": str(graph.id), "name": graph.name} for graph in self.graphs
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
        _validate_write(graph, enabling=graph.enabled)
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
        _validate_write(graph, enabling=enabling)
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
        _validate_write(graph, written=node)
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
        _validate_write(graph, written=node)
    return node


def remove_node(session: Session, graph: AgentGraph, node: AgentGraphNode) -> None:
    """Remove the node and every edge into or out of it."""
    with session.no_autoflush:
        for edge in [e for e in graph.edges if node.id in _ends(e)]:
            graph.edges.remove(edge)
            session.delete(edge)
        graph.nodes.remove(node)
        session.delete(node)
        _validate_write(graph)


def add_edge(
    session: Session, graph: AgentGraph, payload: AgentGraphEdgeCreate
) -> AgentGraphEdge:
    with session.no_autoflush:
        edge = AgentGraphEdge.model_validate(
            payload.model_dump(by_alias=True) | {"agent_graph_id": graph.id}
        )
        _ensure_edge_fits(graph, edge)
        graph.edges.append(edge)
        _validate_write(graph)
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
        _validate_write(graph)
    return edge


def remove_edge(session: Session, graph: AgentGraph, edge: AgentGraphEdge) -> None:
    with session.no_autoflush:
        graph.edges.remove(edge)
        session.delete(edge)
        _validate_write(graph)


def update_agent(
    session: Session, agent: Agent, payload: AgentUpdate, router: Router
) -> Agent:
    """Apply ``payload`` to ``agent`` unless that breaks a graph using the agent.

    Disabling is rejected while an enabled graph uses the agent; enabling is always
    allowed. A change to the agent's arguments, response model or type is rejected
    if it breaks a structural rule of an enabled graph using the agent. Raises
    `ValueError` for an update the agent itself rejects.
    """
    was_enabled = agent.enabled
    contract = _contract(agent)
    with session.no_autoflush:
        graphs = _graphs_using(session, agent, enabled_only=True)
        before = {graph.id: _structure_violations(graph) for graph in graphs}
        agent.apply_update(payload, router)
        if was_enabled and not agent.enabled and graphs:
            raise _in_use(
                f"The agent '{agent.name}' runs in enabled agent graphs; disable "
                "them first, or replace the agent in their nodes.",
                graphs,
                kind="enabled agent graph",
            )
        if _contract(agent) != contract:
            _ensure_nothing_breaks(agent, graphs, before)
    return agent


def delete_agent(session: Session, agent: Agent) -> None:
    """Delete ``agent`` unless a graph, enabled or a draft, has a node that runs it."""
    graphs = _graphs_using(session, agent, enabled_only=False)
    if graphs:
        raise _in_use(
            f"The agent '{agent.name}' runs in agent graphs; remove its nodes from "
            "them, or delete those graphs, first.",
            graphs,
            kind="agent graph",
        )
    session.delete(agent)


def _contract(agent: Agent) -> tuple[object, ...]:
    """What graphs rely on in ``agent``: its arguments, response model and type."""
    arguments = sorted((argument.name, argument.type) for argument in agent.arguments)
    return agent.type, agent.response_model, tuple(arguments)


def _ensure_nothing_breaks(
    agent: Agent,
    graphs: Sequence[AgentGraph],
    before: dict[uuid.UUID, list[str]],
) -> None:
    """Reject the change to ``agent`` if it adds a violation to one of ``graphs``.

    Violations a graph already had don't block it.
    """
    broken: list[AgentGraph] = []
    found: list[str] = []
    for graph in graphs:
        added = [
            violation
            for violation in _structure_violations(graph)
            if violation not in before[graph.id]
        ]
        if added:
            broken.append(graph)
            found += [f"{graph.name}: {violation}" for violation in added]
    if broken:
        raise AgentGraphError(
            status.HTTP_409_CONFLICT,
            f"The change to the agent '{agent.name}' would break enabled agent "
            "graphs; fix the violations, or disable those graphs first.",
            found,
            broken,
        )


def _in_use(
    message: str, graphs: Sequence[AgentGraph], *, kind: str
) -> AgentGraphError:
    return AgentGraphError(
        status.HTTP_409_CONFLICT,
        message,
        [f"The {kind} '{graph.name}' runs the agent." for graph in graphs],
        graphs,
    )


def _graphs_using(
    session: Session, agent: Agent, *, enabled_only: bool
) -> list[AgentGraph]:
    """The graphs with a node that runs ``agent``, by name."""
    query = (
        select(AgentGraph)
        .where(
            col(AgentGraph.id).in_(
                select(AgentGraphNode.agent_graph_id).where(
                    AgentGraphNode.agent_id == agent.id
                )
            )
        )
        .order_by(col(AgentGraph.name))
    )
    if enabled_only:
        query = query.where(col(AgentGraph.enabled).is_(True))
    return list(session.exec(query).all())


def violations(graph: AgentGraph) -> list[str]:
    """Why ``graph`` may not be enabled: disabled node agents and broken structure."""
    return _agent_violations(graph.nodes) + _structure_violations(graph)


def _structure_violations(graph: AgentGraph) -> list[str]:
    outputs, broken = _output_models(graph)
    return [
        *_entry_violations(graph),
        *_reachability_violations(graph),
        *_else_violations(graph),
        *_cycle_violations(graph),
        *broken,
        *_condition_violations(graph, outputs),
        *_input_map_violations(graph, outputs),
        *_binding_violations(graph, outputs),
        *_recovery_violations(graph),
    ]


def _recovery_violations(graph: AgentGraph) -> list[str]:
    """The default recovery graph must be able to run recoveries."""
    if not graph.is_default_recovery:
        return []
    return [
        f"Recoveries can't run the default recovery graph: {gap}, or don't make it "
        "the default."
        for gap in graph.recovery_gaps()
    ]


type OutputModels = dict[str, type[BaseModel] | None]


def _output_models(graph: AgentGraph) -> tuple[OutputModels, list[str]]:
    """Each node's output model by key, and why the ones missing can't load."""
    models: OutputModels = {}
    broken: list[str] = []
    for node in graph.nodes:
        try:
            models[node.key] = node.agent.output_model()
        except (ImportError, ValueError) as exc:
            broken.append(
                f"Node '{node.key}' runs the agent '{node.agent.name}', whose "
                f"response model can't be loaded ({exc}); fix the agent's "
                "response model."
            )
    return models, broken


def _condition_violations(graph: AgentGraph, outputs: OutputModels) -> list[str]:
    """Conditions compare fields of their source's output model, as those fields'
    types allow."""
    nodes = {node.id: node for node in graph.nodes}
    found: list[str] = []
    unroutable: dict[str, AgentGraphNode] = {}
    for edge in graph.edges:
        source = nodes[edge.source_node_id]
        if edge.condition is None or source.key not in outputs:
            continue
        model = outputs[source.key]
        if model is None:
            unroutable[source.key] = source
            continue
        where = (
            f"The condition on the edge '{source.key}' → "
            f"'{nodes[edge.target_node_id].key}'"
        )
        for compare in _compares(CONDITION.validate_python(edge.condition)):
            try:
                annotation = field_type(model, compare.field)
            except FieldPathError as exc:
                found.append(
                    f"{where} compares '{compare.field}', but {exc}; compare a "
                    "field of the source node's output."
                )
                continue
            error = comparison_error(annotation, compare.op, compare.value)
            if error is not None:
                found.append(
                    f"{where} compares '{compare.field}' with '{compare.op}', but "
                    f"{error}; fix the operator or the value."
                )
    return [
        f"Node '{node.key}' has conditional outgoing edges, but its agent "
        f"'{node.agent.name}' has no output model to evaluate them on; give the "
        "agent a response model or remove the conditions."
        for node in unroutable.values()
    ] + found


def _input_map_violations(graph: AgentGraph, outputs: OutputModels) -> list[str]:
    """Input maps map arguments of their node's agent to paths that resolve."""
    found: list[str] = []
    for node in graph.nodes:
        arguments = [argument.name for argument in node.agent.arguments]
        for name, path in node.input_map.items():
            where = f"Node '{node.key}' maps '{name}'"
            if name not in arguments:
                found.append(
                    f"{where}, which isn't an argument of its agent "
                    f"'{node.agent.name}' (arguments: {', '.join(arguments) or 'none'}); "
                    "map one of its arguments."
                )
            error = _path_error(graph, outputs, path)
            if error is not None:
                found.append(f"{where} to '{path}', but {error}.")
    return found


def _path_error(graph: AgentGraph, outputs: OutputModels, path: str) -> str | None:
    """Why an input map ``path`` doesn't resolve, if it doesn't."""
    source, _, fields = path.partition(".")
    if not fields:
        return (
            f"a path is '{RECOVERY_INPUTS_SOURCE}.<field>' or '<node_key>.<field>'; "
            "write it in one of those forms"
        )
    if source == RECOVERY_INPUTS_SOURCE:
        model: type[BaseModel] | None = RecoveryInputs
    elif source not in {node.key for node in graph.nodes}:
        return (
            f"the graph has no node '{source}'; map from one of its nodes or "
            f"from '{RECOVERY_INPUTS_SOURCE}'"
        )
    elif source not in outputs:
        return None
    else:
        model = outputs[source]
    if model is None:
        return (
            f"node '{source}' has no output model; give its agent a response "
            "model or map from elsewhere"
        )
    try:
        _ = field_type(model, fields)
    except FieldPathError as exc:
        return f"{exc}; map an existing field"
    return None


def _binding_violations(graph: AgentGraph, outputs: OutputModels) -> list[str]:
    """Every argument binds on every path into its node: from the recovery inputs,
    from its input map, or by name from a node that runs before on every path."""
    entry = _entry(graph)
    if entry is None:
        return []
    keys = {node.id: node.key for node in graph.nodes}
    dominators = _dominators(graph, entry.id)
    found: list[str] = []
    for node in graph.nodes:
        if node.id not in dominators:
            continue
        before = {keys[d] for d in dominators[node.id] - {node.id}}
        for argument in node.agent.arguments:
            path = node.input_map.get(argument.name)
            if path is not None:
                source = path.partition(".")[0]
                from_node = source != RECOVERY_INPUTS_SOURCE and source in keys.values()
                if from_node and source not in before:
                    found.append(
                        f"Node '{node.key}' maps '{argument.name}' to '{path}', but "
                        f"node '{source}' doesn't run before it on every path; map "
                        "it from a node that does, or from the recovery inputs."
                    )
                continue
            if argument.name in RecoveryInputs.model_fields:
                continue
            outputting = {
                key
                for key, model in outputs.items()
                if model is not None and argument.name in model.model_fields
            }
            if outputting & before:
                continue
            elsewhere = sorted(outputting - before)
            hint = (
                f" Only {', '.join(f'node {k!r}' for k in elsewhere)} outputs it, "
                "which doesn't run before it on every path."
                if elsewhere
                else ""
            )
            found.append(
                f"Node '{node.key}' can start without its argument "
                f"'{argument.name}': it isn't a recovery input, and no node that "
                f"runs before it on every path outputs it.{hint} Map it in the "
                "node's input_map."
            )
    return found


def _dominators(graph: AgentGraph, entry: uuid.UUID) -> dict[uuid.UUID, set[uuid.UUID]]:
    """For each node reachable from ``entry``, the nodes on every path to it,
    itself included; conditions are ignored."""
    reached = _reachable(graph, entry)
    predecessors: dict[uuid.UUID, list[uuid.UUID]] = {n: [] for n in reached}
    for edge in graph.edges:
        if edge.source_node_id in reached and edge.target_node_id in reached:
            predecessors[edge.target_node_id].append(edge.source_node_id)
    dominators = {n: set(reached) for n in reached}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for node in reached - {entry}:
            common = set.intersection(*(dominators[p] for p in predecessors[node]))
            if common | {node} != dominators[node]:
                dominators[node] = common | {node}
                changed = True
    return dominators


def _compares(condition: Condition) -> Iterator[Compare]:
    match condition:
        case Compare():
            yield condition
        case AllOf(all=conditions) | AnyOf(any=conditions):
            for inner in conditions:
                yield from _compares(inner)
        case Not(not_=inner):
            yield from _compares(inner)


def _agent_violations(nodes: Sequence[AgentGraphNode]) -> list[str]:
    return [
        f"Node '{node.key}' runs the disabled agent '{node.agent.name}'; enable "
        "the agent or use another one."
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
    graph: AgentGraph,
    *,
    enabling: bool = False,
    written: AgentGraphNode | None = None,
) -> None:
    """Reject the write if it leaves an enabled ``graph`` invalid.

    Enabling checks every node agent; a write to an enabled graph checks the agent
    of the ``written`` node. Either way the structure must pass.
    """
    graph.updated_at = datetime.now(UTC)
    if not graph.enabled:
        return
    if enabling:
        found = violations(graph)
    else:
        found = _agent_violations([] if written is None else [written])
        found += _structure_violations(graph)
    if found:
        action = "can't be enabled" if enabling else "would no longer be valid"
        raise AgentGraphError(
            status.HTTP_400_BAD_REQUEST,
            f"The agent graph '{graph.name}' {action}; fix the violations, or "
            "disable the graph to edit it as a draft.",
            found,
        )


def _entry(graph: AgentGraph) -> AgentGraphNode | None:
    """The entry node, if the graph has exactly one."""
    entries = [node for node in graph.nodes if node.is_entry]
    return entries[0] if len(entries) == 1 else None


def _entry_violations(graph: AgentGraph) -> list[str]:
    """Exactly one entry node, with no incoming edges."""
    entries = [node for node in graph.nodes if node.is_entry]
    if not entries:
        return ["The graph has no entry node; mark one node as the entry."]
    if len(entries) > 1:
        keys = ", ".join(f"'{node.key}'" for node in entries)
        return [f"The graph has {len(entries)} entry nodes ({keys}); keep exactly one."]
    entry = entries[0]
    incoming = [e for e in graph.edges if e.target_node_id == entry.id]
    if incoming:
        return [
            (
                f"The entry node '{entry.key}' has {len(incoming)} incoming "
                "edge(s); a run only starts there, so remove them."
            )
        ]
    return []


def _reachability_violations(graph: AgentGraph) -> list[str]:
    """Every node is reachable from the entry, ignoring conditions."""
    entry = _entry(graph)
    if entry is None:
        return []
    reached = _reachable(graph, entry.id)
    return [
        f"Node '{node.key}' is unreachable from the entry node "
        f"'{entry.key}'; add an edge into it or remove it."
        for node in graph.nodes
        if node.id not in reached
    ]


def _else_violations(graph: AgentGraph) -> list[str]:
    """At most one unconditional outgoing edge per node, evaluated last."""
    found: list[str] = []
    for node in graph.nodes:
        outgoing = [e for e in graph.edges if e.source_node_id == node.id]
        unconditional = [e for e in outgoing if e.condition is None]
        if len(unconditional) > 1:
            found.append(
                f"Node '{node.key}' has {len(unconditional)} unconditional "
                "outgoing edges; only the first could ever run, so keep one "
                "and give the others conditions."
            )
        elif unconditional and any(
            e.priority > unconditional[0].priority for e in outgoing
        ):
            found.append(
                f"The unconditional edge of node '{node.key}' must have the "
                "lowest priority (the highest number), since edges after it are "
                "never evaluated."
            )
    return found


def _cycle_violations(graph: AgentGraph) -> list[str]:
    """Cycles only with ``max_node_executions``, which has a maximum."""
    found: list[str] = []
    limit = graph.max_node_executions
    if limit is not None and limit > MAX_NODE_EXECUTIONS:
        found.append(
            f"max_node_executions is {limit}, above the maximum of "
            f"{MAX_NODE_EXECUTIONS}; lower it."
        )
    if limit is None and _has_cycle(graph):
        found.append(
            "The graph has a cycle; set max_node_executions to bound its runs."
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


def _rejected(message: str) -> AgentGraphError:
    return AgentGraphError(status.HTTP_400_BAD_REQUEST, message, (message,))


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
        raise _rejected(
            f"An agent graph named '{name}' already exists; choose another name."
        )


def _ensure_unique_key(graph: AgentGraph, key: str, node_id: uuid.UUID | None) -> None:
    if any(n.key == key and n.id != node_id for n in graph.nodes):
        raise _rejected(
            f"The graph already has a node '{key}'; node keys are unique in a graph."
        )


def _ensure_edge_fits(graph: AgentGraph, edge: AgentGraphEdge) -> None:
    """Edges stay inside their graph and are unique; checked even in drafts, since
    the database enforces it too."""
    nodes = {n.id for n in graph.nodes}
    for end in _ends(edge):
        if end not in nodes:
            raise _rejected(
                f"Node {end} isn't a node of the graph '{graph.name}'; edges stay "
                "inside one graph."
            )
    for other in _others(graph, edge):
        if other.source_node_id != edge.source_node_id:
            continue
        if other.target_node_id == edge.target_node_id:
            raise _rejected(
                "An edge between these nodes already exists; change that edge instead."
            )
        if other.priority == edge.priority:
            raise _rejected(
                f"The source node already has an edge with priority {edge.priority}; "
                "give each outgoing edge its own priority."
            )


def _drop_nulls(data: dict[str, Any], *, keep: set[str]) -> dict[str, Any]:
    """Drop explicit nulls, except for fields where null is a value."""
    return {k: v for k, v in data.items() if v is not None or k in keep}


def _others(graph: AgentGraph, edge: AgentGraphEdge) -> Iterator[AgentGraphEdge]:
    return (e for e in graph.edges if e.id != edge.id)


def _ends(edge: AgentGraphEdge) -> tuple[uuid.UUID, uuid.UUID]:
    return edge.source_node_id, edge.target_node_id
