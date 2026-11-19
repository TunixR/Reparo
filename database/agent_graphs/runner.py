"""Reparo's agent graph runner: one path through an agent graph, a node at a time."""

from __future__ import annotations

from collections.abc import Mapping
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
class LoadedNode:
    """A node of a loaded agent graph: its key in the graph and the agent it runs."""

    key: str
    agent: Agent


@dataclass(frozen=True, slots=True)
class LoadedEdge:
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class LoadedAgentGraph:
    """An agent graph with every row it needs loaded, ready to run."""

    entry: LoadedNode
    nodes: tuple[LoadedNode, ...]
    edges: tuple[LoadedEdge, ...] = ()

    @classmethod
    def single(cls, key: str, agent: Agent) -> LoadedAgentGraph:
        node = LoadedNode(key=key, agent=agent)
        return cls(entry=node, nodes=(node,))

    @property
    def node_limit(self) -> int:
        """How many node executions a run may take."""
        return len(self.nodes)

    def next_node(self, node: LoadedNode) -> LoadedNode | None:
        """The node ``node``'s outgoing edge leads to, or ``None`` at the end of the path."""
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
    node: LoadedNode | None = graph.entry
    terminal_node: str | None = None
    output: BaseModel | None = None
    nodes_run = 0
    while node is not None:
        if nodes_run >= graph.node_limit:
            return GraphOutcome(terminal_node, output, nodes_run, "max_node_executions")
        _clean_state(state)
        run = await node.agent.run_node(state, dict(context.recovery_inputs))
        context.outputs[node.key] = run.output
        terminal_node, output = node.key, run.output
        nodes_run += 1
        node = graph.next_node(node)
    return GraphOutcome(terminal_node, output, nodes_run)


def _clean_state(state: dict[str, Any]) -> None:
    """Drop what the previous node's Strands event loop and tracing left behind."""
    for key in [
        k
        for k in state
        if k in _LEFTOVER_STATE_KEYS or k.startswith(_LEFTOVER_STATE_PREFIX)
    ]:
        del state[key]
