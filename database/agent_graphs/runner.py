"""Reparo's agent graph runner: one path through an agent graph, a node at a time."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from database.agents.models import Agent

AGENT_GRAPH_CONTEXT_KEY = "agent_graph"
RECOVERY_INPUTS_SOURCE = "recovery"

_LEFTOVER_STATE_KEYS = ("request_state", "agent", "parent_trace_id")
_LEFTOVER_STATE_PREFIX = "event_loop_"
_MISSING = object()

type FailureReason = Literal["max_node_executions", "execution_timeout"]


@dataclass(frozen=True, slots=True)
class NodeRun:
    """What an agent produces as a node of an agent graph."""

    output: BaseModel
    """The node's structured output, an instance of the agent's `output_model()`."""


@dataclass(frozen=True, slots=True)
class GraphNode:
    key: str
    agent: Agent
    input_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    priority: int = 0
    condition: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class LoadedAgentGraph:
    """An agent graph with every row it needs loaded, ready to run.

    Iterating it walks the run's path: the entry node, the one node without
    incoming edges, then each node's unconditional outgoing edge until none is left.
    """

    nodes: tuple[GraphNode, ...]
    max_node_executions: int
    """A cap on node executions, so a run of a cyclic graph always ends."""
    edges: tuple[GraphEdge, ...] = ()
    execution_timeout_s: float | None = None
    node_timeout_s: float | None = None

    @classmethod
    def single(cls, key: str, agent: Agent) -> LoadedAgentGraph:
        return cls(nodes=(GraphNode(key=key, agent=agent),), max_node_executions=1)

    def __iter__(self) -> Iterator[GraphNode]:
        node: GraphNode | None = self._entry()
        while node is not None:
            yield node
            node = self._next(node)

    def _entry(self) -> GraphNode:
        targets = {edge.target for edge in self.edges}
        entry = next((n for n in self.nodes if n.key not in targets), None)
        if entry is None:
            raise ValueError("The agent graph has no node without incoming edges.")
        return entry

    def _next(self, node: GraphNode) -> GraphNode | None:
        outgoing = sorted(
            (e for e in self.edges if e.source == node.key and e.condition is None),
            key=lambda e: e.priority,
        )
        if not outgoing:
            return None
        return next((n for n in self.nodes if n.key == outgoing[0].target), None)


@dataclass(slots=True)
class AgentGraphContext:
    """The run's own part of the shared invocation state."""

    recovery_inputs: Mapping[str, Any]
    nodes: Mapping[str, GraphNode] = field(default_factory=dict)
    outputs: dict[str, BaseModel] = field(default_factory=dict)

    def bind(self, node: GraphNode) -> dict[str, Any]:
        """The node agent's arguments: from its ``input_map``, else by name.

        By name looks in the recovery inputs, then in the outputs from the most
        recent back. An argument found nowhere is left out.
        """
        inputs: dict[str, Any] = {}
        for argument in node.agent.arguments:
            name = argument.name
            if name in node.input_map:
                inputs[name] = self._resolve(node, name, node.input_map[name])
            elif name in self.recovery_inputs:
                inputs[name] = self.recovery_inputs[name]
            else:
                for output in reversed(self.outputs.values()):
                    if name in type(output).model_fields:
                        inputs[name] = getattr(output, name)
                        break
        return inputs

    def record(self, key: str, output: BaseModel) -> None:
        _ = self.outputs.pop(key, None)
        self.outputs[key] = output

    def _resolve(self, node: GraphNode, argument: str, path: str) -> Any:
        source, *fields = path.split(".")
        value: Any = (
            self.recovery_inputs
            if source == RECOVERY_INPUTS_SOURCE
            else self.outputs.get(source, _MISSING)
        )
        for name in fields:
            if isinstance(value, BaseModel) and name in type(value).model_fields:
                value = getattr(value, name)
            elif isinstance(value, Mapping) and name in value:
                value = value[name]  # pyright: ignore[reportUnknownVariableType]
            else:
                value = _MISSING
                break
        if value is _MISSING or not fields:
            raise ValueError(
                f"Node '{node.key}' maps argument '{argument}' to '{path}', "
                "which has no value."
            )
        return value  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True, slots=True)
class GraphOutcome:
    """How a run ended: the last executed node, and why it failed if it did."""

    terminal_node: str | None
    output: BaseModel | None
    nodes_run: int
    failure_reason: FailureReason | None = None


async def run_graph(graph: LoadedAgentGraph, state: dict[str, Any]) -> GraphOutcome:
    """Run ``graph`` from its entry node until no outgoing edge is left.

    Nodes are awaited in the caller's task, so cancelling the caller cancels the
    running node. Anything a node raises propagates unchanged, including the
    `TimeoutError` of a node that runs past ``node_timeout_s``. ``state`` holds an
    `AgentGraphContext` under `AGENT_GRAPH_CONTEXT_KEY` and is shared, never copied,
    by every node.
    """
    context: AgentGraphContext = state[AGENT_GRAPH_CONTEXT_KEY]
    context.nodes = {node.key: node for node in graph.nodes}
    deadline = (
        None
        if graph.execution_timeout_s is None
        else monotonic() + graph.execution_timeout_s
    )
    terminal_node: str | None = None
    output: BaseModel | None = None
    nodes_run = 0
    for node in graph:
        if nodes_run >= graph.max_node_executions:
            return GraphOutcome(terminal_node, output, nodes_run, "max_node_executions")
        if deadline is not None and monotonic() > deadline:
            return GraphOutcome(terminal_node, output, nodes_run, "execution_timeout")
        _clean_state(state)
        inputs = context.bind(node)
        async with asyncio.timeout(graph.node_timeout_s):
            run = await node.agent.run_node(state, inputs)
        context.record(node.key, run.output)
        terminal_node, output = node.key, run.output
        nodes_run += 1
    return GraphOutcome(terminal_node, output, nodes_run)


def _clean_state(state: dict[str, Any]) -> None:
    """Drop what the previous node's Strands event loop and tracing left behind."""
    for key in [
        k
        for k in state
        if k in _LEFTOVER_STATE_KEYS or k.startswith(_LEFTOVER_STATE_PREFIX)
    ]:
        del state[key]
