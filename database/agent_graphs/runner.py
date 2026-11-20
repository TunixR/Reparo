"""The runtime state of an agent graph run, shared by the graph and its nodes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from database.agent_graphs.models import AgentGraphNode

AGENT_GRAPH_CONTEXT_KEY = "agent_graph"
RECOVERY_INPUTS_SOURCE = "recovery"

_LEFTOVER_STATE_KEYS = ("request_state", "agent", "parent_trace_id")
_LEFTOVER_STATE_PREFIX = "event_loop_"

type FailureReason = Literal["max_node_executions", "execution_timeout"]


class NodeCancelledError(RuntimeError):
    """A node's agent was disabled after the run started, so the node didn't run."""


@dataclass(frozen=True, slots=True)
class NodeRun:
    """What an agent produces as a node of an agent graph."""

    output: BaseModel
    """The node's structured output, an instance of the agent's `output_model()`."""


@dataclass(slots=True)
class AgentGraphContext:
    """The run's own part of the shared invocation state."""

    recovery_inputs: Mapping[str, Any]
    nodes: Mapping[str, AgentGraphNode] = field(default_factory=dict)
    outputs: dict[str, BaseModel] = field(default_factory=dict)

    def record(self, key: str, output: BaseModel) -> None:
        """Store a node's output, the latest execution last."""
        _ = self.outputs.pop(key, None)
        self.outputs[key] = output

    def get(self, path: str) -> Any:
        """The value at ``recovery.<field>`` or ``<node_key>.<field>``, dotted.

        A bare name is a recovery input, else that field of the most recent output
        that has it. ``None`` when nothing has the value.
        """
        source, *fields = path.split(".")
        if not fields:
            if source in self.recovery_inputs:
                return self.recovery_inputs[source]
            return next(
                (
                    getattr(output, source)
                    for output in reversed(self.outputs.values())
                    if source in type(output).model_fields
                ),
                None,
            )
        value: Any = (
            self.recovery_inputs
            if source == RECOVERY_INPUTS_SOURCE
            else self.outputs.get(source)
        )
        for name in fields:
            if isinstance(value, BaseModel) and name in type(value).model_fields:
                value = getattr(value, name)
            elif isinstance(value, Mapping) and name in value:
                value = value[name]  # pyright: ignore[reportUnknownVariableType] - JSON of unknown shape
            else:
                return None
        return value  # pyright: ignore[reportUnknownVariableType] - JSON of unknown shape


@dataclass(frozen=True, slots=True)
class GraphOutcome:
    """How a run ended: the last executed node, and why it failed if it did."""

    terminal_node: str | None
    output: BaseModel | None
    nodes_run: int
    failure_reason: FailureReason | None = None


def clean_state(state: dict[str, Any]) -> None:
    """Drop what the previous node's Strands event loop and tracing left behind."""
    for key in [
        k
        for k in state
        if k in _LEFTOVER_STATE_KEYS or k.startswith(_LEFTOVER_STATE_PREFIX)
    ]:
        del state[key]
