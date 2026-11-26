import asyncio
import enum
import logging
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from time import monotonic
from typing import Any, NoReturn

from pydantic import BaseModel
from sqlalchemy import Index, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import object_session, validates
from sqlmodel import Field, Relationship, Session, SQLModel, col, select

from agent_tools.utils import SET_CONTINUATION_FN_MODULE
from database.agent_graphs.conditions import CONDITION, Condition, ConditionType
from database.agent_graphs.runner import (
    AGENT_GRAPH_CONTEXT_KEY,
    AgentGraphContext,
    GraphOutcome,
    NodeCancelledError,
    clean_state,
)
from database.agents.behaviours import behaviour_of, is_gui_agent
from database.agents.models import Agent
from observability import attributes as attrs
from observability.events import brief, error_type, log_event
from observability.instrumentation.agent_graph import (
    AgentGraphRunTelemetry,
    agent_graph_run,
)

logger = logging.getLogger(__name__)


class AgentGraphFlag(str, enum.Enum):
    """Something an agent graph can't be used for."""

    NOT_RUNNABLE = "NOT_RUNNABLE"
    NOT_RECOVERY_FIT = "NOT_RECOVERY_FIT"


def _reports_success(node: "AgentGraphNode") -> bool:
    """Whether the node's output model has a boolean ``success``."""
    model = behaviour_of(node.agent).output_class()
    success = None if model is None else model.model_fields.get("success")
    return success is not None and success.annotation is bool


class AgentGraph(SQLModel, table=True):
    """A pipeline of agents as nodes, run one at a time along the edges."""

    __table_args__ = (
        Index(
            "uq_agentgraph_default_recovery",
            "is_default_recovery",
            unique=True,
            postgresql_where=text("is_default_recovery"),
            sqlite_where=text("is_default_recovery"),
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, description="Name of the agent graph.")
    description: str = Field(description="What the agent graph does.")
    enabled: bool = Field(default=False, description="Whether the graph may run.")
    is_default_recovery: bool = Field(
        default=False,
        description="Whether recoveries run this graph; at most one graph is.",
    )
    max_node_executions: int | None = Field(
        default=None,
        gt=0,
        description="Cap on node executions per run; unset caps at the node count.",
    )
    execution_timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Soft run deadline, checked before each node; unset is off.",
    )
    node_timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Hard deadline for one node, which cancels it; unset is off.",
    )

    nodes: list["AgentGraphNode"] = Relationship(
        back_populates="agent_graph",
        cascade_delete=True,
        sa_relationship_kwargs={"lazy": "selectin"},
    )
    edges: list["AgentGraphEdge"] = Relationship(
        back_populates="agent_graph", cascade_delete=True
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def default_recovery_id(cls, session: Session) -> uuid.UUID | None:
        """The id of the enabled graph recoveries run, if there is one."""
        return session.exec(
            select(cls.id).where(
                col(cls.is_default_recovery).is_(True), col(cls.enabled).is_(True)
            )
        ).first()

    @property
    def entry(self) -> "AgentGraphNode":
        entry = next((node for node in self.nodes if node.is_entry), None)
        if entry is None:
            raise ValueError(f"The agent graph '{self.name}' has no entry node.")
        return entry

    @property
    def flags(self) -> list[AgentGraphFlag]:
        """What the graph can't be used for."""
        flags: list[AgentGraphFlag] = []
        if not self.can_run:
            flags.append(AgentGraphFlag.NOT_RUNNABLE)
        if self.recovery_gaps():
            flags.append(AgentGraphFlag.NOT_RECOVERY_FIT)
        return flags

    @property
    def can_run(self) -> bool:
        """Whether every node's agent is enabled, as the database has it now.

        The agents loaded with the graph keep the flag they had when it was loaded,
        so a running graph wouldn't see an agent disabled since.
        """
        session = object_session(self)
        if session is None:
            return all(node.agent.enabled for node in self.nodes)
        disabled = session.scalar(
            select(func.count())
            .select_from(Agent)
            .where(
                col(Agent.id).in_({node.agent_id for node in self.nodes}),
                col(Agent.enabled).is_(False),
            )
        )
        return not disabled

    def recovery_gaps(self) -> list[str]:
        """Why recoveries can't run the graph; empty when they can."""
        gaps: list[str] = []
        if not any(is_gui_agent(node.agent) for node in self.nodes):
            gaps.append(
                "no node runs a GUI agent, so nothing acts on the robot's desktop; "
                "add one"
            )
        if not any(
            tool.fn_module == SET_CONTINUATION_FN_MODULE
            for node in self.nodes
            for tool in node.agent.get_tools()
        ):
            gaps.append(
                "no node's agent has the set_continuation tool, so a recovery can't "
                "say where the robot continues; give it to one"
            )
        if not any(
            _reports_success(node) for node in self.nodes if self._can_end(node)
        ):
            gaps.append(
                "no node that can end the graph outputs a boolean 'success', so every "
                "recovery is reported as not recovered; add 'success: bool' to the "
                "output model of one"
            )
        return gaps

    def _can_end(self, node: "AgentGraphNode") -> bool:
        """Whether a run can end at ``node``: it has no unconditional outgoing edge."""
        return not any(
            edge.condition is None
            for edge in self.edges
            if edge.source_node_id == node.id
        )

    @property
    def node_limit(self) -> int:
        """``max_node_executions``, or the node count, so every run ends."""
        return self.max_node_executions or len(self.nodes)

    async def run(self, state: dict[str, Any]) -> GraphOutcome:
        """Run from the entry node until no outgoing edge of a node holds.

        Nodes are awaited in the caller's task, so cancelling the caller cancels the
        running node. Anything a node raises propagates unchanged, including the
        `TimeoutError` of a node that runs past ``node_timeout_s``. Once an agent of
        the graph is disabled during the run, the next node raises
        `NodeCancelledError` instead of running. ``state`` holds an
        `AgentGraphContext` under `AGENT_GRAPH_CONTEXT_KEY` and is shared, never
        copied, by every node.
        """
        with agent_graph_run(self.id, self.name) as telemetry:
            outcome = await self._run(state, telemetry)
            telemetry.completed(outcome.failure_reason)
            return outcome

    async def _run(
        self, state: dict[str, Any], telemetry: AgentGraphRunTelemetry
    ) -> GraphOutcome:
        context: AgentGraphContext = state[AGENT_GRAPH_CONTEXT_KEY]
        context.nodes = {node.key: node for node in self.nodes}
        deadline = (
            None
            if self.execution_timeout_s is None
            else monotonic() + self.execution_timeout_s
        )
        terminal_node: str | None = None
        output: BaseModel | None = None
        nodes_run = 0
        node: AgentGraphNode | None = self.entry
        entered_by: AgentGraphEdge | None = None
        while node is not None:
            if nodes_run >= self.node_limit:
                return GraphOutcome(
                    terminal_node, output, nodes_run, "max_node_executions"
                )
            if deadline is not None and monotonic() > deadline:
                return GraphOutcome(
                    terminal_node, output, nodes_run, "execution_timeout"
                )
            clean_state(state)
            with (
                telemetry.node(nodes_run, node.key, node.agent_id),
                self._node_trace(context, nodes_run, node, entered_by),
            ):
                if not self.can_run:
                    self._cancel(node)
                async with asyncio.timeout(self.node_timeout_s):
                    run = await behaviour_of(node.agent).run_node(state, node.input_map)
            context.record(node.key, run.output)
            terminal_node, output = node.key, run.output
            nodes_run += 1
            entered_by = node.next_edge(run.output)
            node = None if entered_by is None else entered_by.target
        return GraphOutcome(terminal_node, output, nodes_run)

    @staticmethod
    def _node_trace(
        context: AgentGraphContext,
        execution_index: int,
        node: "AgentGraphNode",
        entered_by: "AgentGraphEdge | None",
    ) -> AbstractContextManager[None]:
        """The node trace of this execution, when the run is traced."""
        if context.trace is None:
            return nullcontext()
        return context.trace.node(execution_index, node.key, entered_by)

    def _cancel(self, node: "AgentGraphNode") -> NoReturn:
        """Log that ``node`` won't run, and raise `NodeCancelledError` to end the run."""
        log_event(
            logger,
            attrs.EVENT_AGENT_GRAPH_NODE_CANCELLED,
            {
                attrs.AGENT_GRAPH_ID: self.id,
                attrs.AGENT_GRAPH_NODE_KEY: node.key,
                attrs.AGENT_ID: node.agent_id,
            },
            logging.WARNING,
        )
        raise NodeCancelledError(
            f"Node '{node.key}' of the agent graph '{self.name}' was cancelled: an "
            "agent of the graph was disabled during the run."
        )


class AgentGraphNode(SQLModel, table=True):
    """An agent's place in an agent graph; one agent may back several nodes."""

    __table_args__ = (UniqueConstraint("agent_graph_id", "key"),)

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    agent_graph_id: uuid.UUID = Field(
        foreign_key="agentgraph.id", ondelete="CASCADE", index=True
    )
    agent_graph: AgentGraph = Relationship(back_populates="nodes")
    key: str = Field(
        description="The node's identity in its graph, and the key of its output."
    )
    agent_id: uuid.UUID = Field(foreign_key="agent.id", ondelete="RESTRICT")
    agent: "Agent" = Relationship(  # pyright: ignore[reportUndefinedVariable]
        back_populates="graph_nodes", sa_relationship_kwargs={"lazy": "joined"}
    )
    is_entry: bool = Field(default=False, description="Whether the run starts here.")
    input_map: dict[str, str] = Field(
        default_factory=dict,
        sa_type=JSONB,
        description=(
            'Argument name to "recovery.<field>" or "<node_key>.<field>"; '
            "unmapped arguments bind by name."
        ),
    )

    outgoing: list["AgentGraphEdge"] = Relationship(
        back_populates="source",
        sa_relationship_kwargs={
            "foreign_keys": "AgentGraphEdge.source_node_id",
            "passive_deletes": True,
            "lazy": "selectin",
        },
    )

    def next_edge(self, output: BaseModel | None) -> "AgentGraphEdge | None":
        """The first outgoing edge, by priority, that holds over ``output``.

        Edges after that one aren't evaluated. ``None`` when no edge holds.
        """
        edges = sorted(self.outgoing, key=lambda edge: edge.priority)
        return next((edge for edge in edges if edge.holds(output)), None)


class AgentGraphEdge(SQLModel, table=True):
    """A step from one node to the next; lower priorities are evaluated first."""

    __table_args__ = (
        UniqueConstraint("source_node_id", "target_node_id"),
        UniqueConstraint("source_node_id", "priority"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    agent_graph_id: uuid.UUID = Field(
        foreign_key="agentgraph.id", ondelete="CASCADE", index=True
    )
    agent_graph: AgentGraph = Relationship(back_populates="edges")
    source_node_id: uuid.UUID = Field(
        foreign_key="agentgraphnode.id", ondelete="CASCADE"
    )
    source: AgentGraphNode = Relationship(
        back_populates="outgoing",
        sa_relationship_kwargs={"foreign_keys": "AgentGraphEdge.source_node_id"},
    )
    target_node_id: uuid.UUID = Field(
        foreign_key="agentgraphnode.id", ondelete="CASCADE"
    )
    target: AgentGraphNode = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "AgentGraphEdge.target_node_id",
            "lazy": "joined",
        }
    )
    priority: int = Field(default=0)
    condition: Condition | None = Field(
        default=None,
        sa_type=ConditionType,
        description="Predicate over the source node's output; null is unconditional.",
    )

    @validates("condition")
    def validate_condition(
        self, _key: str, condition: Condition | Mapping[str, Any] | None
    ) -> Condition | Mapping[str, Any] | None:
        """Reject a malformed condition where the edge is built, not at flush.

        SQLModel keeps the assigned value, so a dict stays a dict until it is
        stored; `ConditionType` loads it back as a `Condition`.
        """
        if condition is not None:
            _ = CONDITION.validate_python(condition)
        return condition

    def holds(self, output: BaseModel | None) -> bool:
        """Whether the run may follow this edge after its source produced ``output``.

        An unconditional edge always holds. A condition over a missing output, or one
        that raises (a missing field, a type error), doesn't, and the error is logged.
        """
        if self.condition is None:
            return True
        if output is None:
            return False
        try:
            condition = CONDITION.validate_python(self.condition)
            return condition.holds(output.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                attrs.EVENT_AGENT_GRAPH_CONDITION_ERROR,
                {
                    attrs.AGENT_GRAPH_ID: self.agent_graph_id,
                    attrs.AGENT_GRAPH_EDGE_ID: self.id,
                    attrs.ERROR_TYPE: error_type(exc),
                    attrs.EXCEPTION_MESSAGE: brief(exc),
                },
                logging.WARNING,
            )
            return False


_TAKES_OVER_DEFAULT_RECOVERY = (
    "Whether recoveries run this graph; it takes over from the previous one."
)


class AgentGraphLimits(SQLModel):
    """The run limits a graph is created or updated with."""

    max_node_executions: int | None = Field(
        default=None,
        gt=0,
        description="Cap on node executions per run; required when the graph has a cycle.",
    )
    execution_timeout_s: float | None = Field(
        default=None, gt=0, description="Soft run deadline in seconds; null is off."
    )
    node_timeout_s: float | None = Field(
        default=None,
        gt=0,
        description="Hard deadline for one node in seconds; null is off.",
    )


class AgentGraphCreate(AgentGraphLimits):
    """Schema for creating an agent graph; it starts as a draft unless enabled."""

    name: str = Field(description="Name of the agent graph.")
    description: str = Field(description="What the agent graph does.")
    enabled: bool = Field(default=False, description="Whether the graph may run.")
    is_default_recovery: bool = Field(
        default=False,
        description=_TAKES_OVER_DEFAULT_RECOVERY,
    )


class AgentGraphUpdate(AgentGraphLimits):
    """Schema for updating an agent graph; a null limit turns that limit off."""

    name: str | None = Field(default=None, description="Name of the agent graph.")
    description: str | None = Field(
        default=None, description="What the agent graph does."
    )
    enabled: bool | None = Field(default=None, description="Whether the graph may run.")
    is_default_recovery: bool | None = Field(
        default=None,
        description=_TAKES_OVER_DEFAULT_RECOVERY,
    )


class AgentGraphNodeCreate(SQLModel):
    """Schema for adding a node to an agent graph."""

    key: str = Field(description="The node's identity in its graph.")
    agent_id: uuid.UUID = Field(description="The agent that runs as this node.")
    is_entry: bool = Field(default=False, description="Whether the run starts here.")
    input_map: dict[str, str] = Field(
        default_factory=dict,
        description='Argument name to "recovery.<field>" or "<node_key>.<field>".',
    )


class AgentGraphNodeUpdate(SQLModel):
    """Schema for changing a node of an agent graph."""

    key: str | None = Field(
        default=None, description="The node's identity in its graph."
    )
    agent_id: uuid.UUID | None = Field(
        default=None, description="The agent that runs as this node."
    )
    is_entry: bool | None = Field(
        default=None, description="Whether the run starts here."
    )
    input_map: dict[str, str] | None = Field(
        default=None,
        description='Argument name to "recovery.<field>" or "<node_key>.<field>".',
    )


class AgentGraphEdgeCreate(SQLModel):
    """Schema for adding an edge between two nodes of the same agent graph."""

    source_node_id: uuid.UUID = Field(description="The node the edge leaves.")
    target_node_id: uuid.UUID = Field(description="The node the edge enters.")
    priority: int = Field(
        default=0, description="Lower priorities are evaluated first."
    )
    condition: Condition | None = Field(
        default=None,
        description="Predicate over the source node's output; null is unconditional.",
    )


class AgentGraphEdgeUpdate(SQLModel):
    """Schema for changing an edge; a null condition makes it unconditional."""

    target_node_id: uuid.UUID | None = Field(
        default=None, description="The node the edge enters."
    )
    priority: int | None = Field(
        default=None, description="Lower priorities are evaluated first."
    )
    condition: Condition | None = Field(
        default=None,
        description="Predicate over the source node's output; null is unconditional.",
    )


class AgentGraphNodeRead(SQLModel):
    id: uuid.UUID
    key: str
    agent_id: uuid.UUID
    is_entry: bool
    input_map: dict[str, str]


class AgentGraphEdgeRead(SQLModel):
    id: uuid.UUID
    source_node_id: uuid.UUID
    target_node_id: uuid.UUID
    priority: int
    condition: Condition | None


class AgentGraphRead(SQLModel):
    """An agent graph with its nodes and edges."""

    id: uuid.UUID
    name: str
    description: str
    enabled: bool
    is_default_recovery: bool
    max_node_executions: int | None
    execution_timeout_s: float | None
    node_timeout_s: float | None
    nodes: list[AgentGraphNodeRead]
    edges: list[AgentGraphEdgeRead]
    created_at: datetime
    updated_at: datetime


class AgentGraphDetail(AgentGraphRead):
    """An agent graph with its nodes, its edges and what it can't be used for."""

    flags: list[AgentGraphFlag] = Field(
        description="What the graph can't be used for, such as recoveries."
    )
