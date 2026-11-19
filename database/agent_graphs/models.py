import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel


class AgentGraph(SQLModel, table=True):
    """A recovery pipeline: agents as nodes, run one at a time along the edges."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, description="Name of the agent graph.")
    description: str = Field(description="What the agent graph recovers.")
    enabled: bool = Field(
        default=False,
        description="Whether recoveries entering at its gateway run this graph.",
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
        back_populates="agent_graph", cascade_delete=True
    )
    edges: list["AgentGraphEdge"] = Relationship(
        back_populates="agent_graph", cascade_delete=True
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


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
    agent: "Agent" = Relationship(back_populates="graph_nodes")  # noqa: F821 # pyright: ignore[reportUndefinedVariable]
    is_entry: bool = Field(default=False, description="Whether the run starts here.")
    input_map: dict[str, str] = Field(
        default_factory=dict,
        sa_type=JSONB,
        description=(
            'Argument name to "recovery.<field>" or "<node_key>.<field>"; '
            "unmapped arguments bind by name."
        ),
    )


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
    target_node_id: uuid.UUID = Field(
        foreign_key="agentgraphnode.id", ondelete="CASCADE"
    )
    priority: int = Field(default=0)
    condition: dict[str, Any] | None = Field(
        default=None,
        sa_type=JSONB,
        description="Predicate over the source node's output; null is unconditional.",
    )
