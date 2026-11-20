import asyncio
import enum
import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Index, UniqueConstraint, text
from sqlalchemy.orm import object_session, validates
from sqlalchemy.dialects.postgresql import JSONB
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
from database.agents.models import Agent, AgentType
from observability import attributes as attrs
from observability.events import brief, error_type, log_event

logger = logging.getLogger(__name__)


class AgentGraphFlag(str, enum.Enum):
    """Something an agent graph can't be used for."""

    NOT_RECOVERY_FIT = "NOT_RECOVERY_FIT"


def _reports_success(node: "AgentGraphNode") -> bool:
    """Whether the node's output model has a boolean ``success``."""
    try:
        model = node.agent.output_model()
    except (ImportError, ValueError):
        return False
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
        return [AgentGraphFlag.NOT_RECOVERY_FIT] if self.recovery_gaps() else []

    def recovery_gaps(self) -> list[str]:
        """Why recoveries can't run the graph; empty when they can."""
        gaps: list[str] = []
        if not any(node.agent.type == AgentType.GuiAgent for node in self.nodes):
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
        `TimeoutError` of a node that runs past ``node_timeout_s``. A node whose agent
        was disabled since the run started raises `NodeCancelledError` before it
        runs. ``state`` holds an `AgentGraphContext` under `AGENT_GRAPH_CONTEXT_KEY`
        and is shared, never copied, by every node.
        """
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
        while node is not None:
            if nodes_run >= self.node_limit:
                return GraphOutcome(
                    terminal_node, output, nodes_run, "max_node_executions"
                )
            if deadline is not None and monotonic() > deadline:
                return GraphOutcome(
                    terminal_node, output, nodes_run, "execution_timeout"
                )
            self._ensure_agent_enabled(node)
            clean_state(state)
            async with asyncio.timeout(self.node_timeout_s):
                run = await node.agent.run_node(state, node.input_map)
            context.record(node.key, run.output)
            terminal_node, output = node.key, run.output
            nodes_run += 1
            node = node.next_node(run.output)
        return GraphOutcome(terminal_node, output, nodes_run)

    def _ensure_agent_enabled(self, node: "AgentGraphNode") -> None:
        """Raise `NodeCancelledError` unless the database has ``node``'s agent enabled."""
        session = object_session(self)
        if session is None:
            raise RuntimeError(f"The agent graph '{self.name}' isn't in a session.")
        if session.scalar(select(Agent.enabled).where(Agent.id == node.agent_id)):
            return
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
            f"Node '{node.key}' of the agent graph '{self.name}' was cancelled: its "
            f"agent '{node.agent.name}' was disabled during the run."
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
    agent: "Agent" = Relationship(  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
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

    def next_node(self, output: BaseModel | None) -> "AgentGraphNode | None":
        """The target of the first outgoing edge, by priority, that holds over ``output``.

        Edges after that one aren't evaluated. ``None`` when no edge holds.
        """
        edges = sorted(self.outgoing, key=lambda edge: edge.priority)
        return next((edge.target for edge in edges if edge.holds(output)), None)


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
        except Exception as exc:
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


class AgentGraphCreate(SQLModel):
    """Schema for creating an agent graph; it starts as a draft unless enabled."""

    name: str = Field(description="Name of the agent graph.")
    description: str = Field(description="What the agent graph does.")
    enabled: bool = Field(default=False, description="Whether the graph may run.")
    is_default_recovery: bool = Field(
        default=False,
        description="Whether recoveries run this graph; it takes over from the previous one.",
    )
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


class AgentGraphUpdate(SQLModel):
    """Schema for updating an agent graph; a null limit turns that limit off."""

    name: str | None = Field(default=None, description="Name of the agent graph.")
    description: str | None = Field(
        default=None, description="What the agent graph does."
    )
    enabled: bool | None = Field(default=None, description="Whether the graph may run.")
    is_default_recovery: bool | None = Field(
        default=None,
        description="Whether recoveries run this graph; it takes over from the previous one.",
    )
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
