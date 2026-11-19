import asyncio
import logging
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from time import monotonic
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Index, UniqueConstraint, text
from sqlalchemy.orm import validates
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, Session, SQLModel, col, select

from database.agent_graphs.conditions import CONDITION, Condition, ConditionType
from database.agent_graphs.runner import (
    AGENT_GRAPH_CONTEXT_KEY,
    AgentGraphContext,
    GraphOutcome,
    clean_state,
)
from observability import attributes as attrs
from observability.events import brief, error_type, log_event

logger = logging.getLogger(__name__)


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
    def node_limit(self) -> int:
        """``max_node_executions``, or the node count, so every run ends."""
        return self.max_node_executions or len(self.nodes)

    async def run(self, state: dict[str, Any]) -> GraphOutcome:
        """Run from the entry node until no outgoing edge of a node holds.

        Nodes are awaited in the caller's task, so cancelling the caller cancels the
        running node. Anything a node raises propagates unchanged, including the
        `TimeoutError` of a node that runs past ``node_timeout_s``. ``state`` holds an
        `AgentGraphContext` under `AGENT_GRAPH_CONTEXT_KEY` and is shared, never
        copied, by every node.
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
            clean_state(state)
            async with asyncio.timeout(self.node_timeout_s):
                run = await node.agent.run_node(state, node.input_map)
            context.record(node.key, run.output)
            terminal_node, output = node.key, run.output
            nodes_run += 1
            node = node.next_node(run.output)
        return GraphOutcome(terminal_node, output, nodes_run)


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
