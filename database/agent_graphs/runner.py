"""Reparo's agent graph runner: one path through an agent graph, a node at a time."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from database.agents.models import Agent

AGENT_GRAPH_CONTEXT_KEY = "agent_graph"

_LEFTOVER_STATE_KEYS = ("request_state", "agent", "parent_trace_id")
_LEFTOVER_STATE_PREFIX = "event_loop_"

type FailureReason = Literal["max_node_executions"]


@dataclass(frozen=True, slots=True)
class NodeRun:
    """What an agent produces as a node of an agent graph."""

    output: BaseModel
    """The node's structured output, an instance of the agent's `output_model()`."""


@dataclass(frozen=True, slots=True)
class AgentGraphNode:
    key: str
    agent: Agent


@dataclass(frozen=True, slots=True)
class AgentGraphEdge:
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class LoadedAgentGraph:
    """An agent graph with every row it needs loaded, ready to run.

    Iterating it walks the run's path: the entry node, the one node without
    incoming edges, then each node's outgoing edge until none is left.
    """

    nodes: tuple[AgentGraphNode, ...]
    max_node_executions: int
    """A cap on node executions, so a run of a cyclic graph always ends."""
    edges: tuple[AgentGraphEdge, ...] = ()

    @classmethod
    def single(cls, key: str, agent: Agent) -> LoadedAgentGraph:
        return cls(nodes=(AgentGraphNode(key=key, agent=agent),), max_node_executions=1)

    def __iter__(self) -> Iterator[AgentGraphNode]:
        node: AgentGraphNode | None = self._entry()
        while node is not None:
            yield node
            node = self._next(node)

    def _entry(self) -> AgentGraphNode:
        targets = {edge.target for edge in self.edges}
        entry = next((n for n in self.nodes if n.key not in targets), None)
        if entry is None:
            raise ValueError("The agent graph has no node without incoming edges.")
        return entry

    def _next(self, node: AgentGraphNode) -> AgentGraphNode | None:
        target = next((e.target for e in self.edges if e.source == node.key), None)
        return next((n for n in self.nodes if n.key == target), None)


@dataclass(slots=True)
class AgentGraphContext:
    """The run's own part of the shared invocation state."""

    recovery_inputs: Mapping[str, Any]
    outputs: dict[str, BaseModel] = field(default_factory=dict)


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
    running node. Anything a node raises propagates unchanged. ``state`` holds an
    `AgentGraphContext` under `AGENT_GRAPH_CONTEXT_KEY` and is shared, never copied,
    by every node.
    """
    context: AgentGraphContext = state[AGENT_GRAPH_CONTEXT_KEY]
    terminal_node: str | None = None
    output: BaseModel | None = None
    nodes_run = 0
    for node in graph:
        if nodes_run >= graph.max_node_executions:
            return GraphOutcome(terminal_node, output, nodes_run, "max_node_executions")
        _clean_state(state)
        run = await node.agent.run_node(state, dict(context.recovery_inputs))
        context.outputs[node.key] = run.output
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
