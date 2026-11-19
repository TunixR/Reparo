"""agent graphs

Adds the agent graph tables: graphs, their nodes (each backed by an agent) and the
edges between nodes. Recoveries run the enabled default recovery graph, so an
existing gateway becomes the one node of a new default recovery graph, which keeps
recoveries running as before.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-19 11:18:09.163742

"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEFAULT_RECOVERY_GRAPH = "Default Recovery"
DEFAULT_RECOVERY_DESCRIPTION = "Runs the gateway agent on every recovery."
GATEWAY_NODE_KEY = "gateway"


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "agentgraph",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("is_default_recovery", sa.Boolean(), nullable=False),
        sa.Column("max_node_executions", sa.Integer(), nullable=True),
        sa.Column("execution_timeout_s", sa.Float(), nullable=True),
        sa.Column("node_timeout_s", sa.Float(), nullable=True),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("updated_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(
        "uq_agentgraph_default_recovery",
        "agentgraph",
        ["is_default_recovery"],
        unique=True,
        postgresql_where=sa.text("is_default_recovery"),
        sqlite_where=sa.text("is_default_recovery"),
    )
    op.create_table(
        "agentgraphnode",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_graph_id", sa.Uuid(), nullable=False),
        sa.Column("key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.Column("is_entry", sa.Boolean(), nullable=False),
        sa.Column("input_map", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_graph_id"], ["agentgraph.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agent.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_graph_id", "key"),
    )
    op.create_index(
        op.f("ix_agentgraphnode_agent_graph_id"),
        "agentgraphnode",
        ["agent_graph_id"],
        unique=False,
    )
    op.create_table(
        "agentgraphedge",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_graph_id", sa.Uuid(), nullable=False),
        sa.Column("source_node_id", sa.Uuid(), nullable=False),
        sa.Column("target_node_id", sa.Uuid(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("condition", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_graph_id"], ["agentgraph.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_node_id"], ["agentgraphnode.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["target_node_id"], ["agentgraphnode.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_node_id", "priority"),
        sa.UniqueConstraint("source_node_id", "target_node_id"),
    )
    op.create_index(
        op.f("ix_agentgraphedge_agent_graph_id"),
        "agentgraphedge",
        ["agent_graph_id"],
        unique=False,
    )

    _create_default_recovery_graph()


def _create_default_recovery_graph() -> None:
    """Make the oldest gateway agent, if any, the one node of the default recovery graph."""
    bind = op.get_bind()
    gateway_id = bind.execute(
        sa.text(
            "SELECT id FROM agent WHERE type = 'GatewayAgent' ORDER BY created_at LIMIT 1"
        )
    ).scalar()
    if gateway_id is None:
        return
    graph_id = uuid.uuid4()
    now = datetime.now(UTC)
    bind.execute(
        sa.text(
            "INSERT INTO agentgraph (id, name, description, enabled, is_default_recovery,"
            " created_at, updated_at)"
            " VALUES (:id, :name, :description, true, true, :now, :now)"
        ),
        {
            "id": graph_id,
            "name": DEFAULT_RECOVERY_GRAPH,
            "description": DEFAULT_RECOVERY_DESCRIPTION,
            "now": now,
        },
    )
    bind.execute(
        sa.text(
            "INSERT INTO agentgraphnode (id, agent_graph_id, key, agent_id, is_entry,"
            " input_map) VALUES (:id, :graph_id, :key, :agent_id, true, '{}'::jsonb)"
        ),
        {
            "id": uuid.uuid4(),
            "graph_id": graph_id,
            "key": GATEWAY_NODE_KEY,
            "agent_id": gateway_id,
        },
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_agentgraphedge_agent_graph_id"), table_name="agentgraphedge")
    op.drop_table("agentgraphedge")
    op.drop_index(op.f("ix_agentgraphnode_agent_graph_id"), table_name="agentgraphnode")
    op.drop_table("agentgraphnode")
    op.drop_index(
        "uq_agentgraph_default_recovery",
        table_name="agentgraph",
        postgresql_where=sa.text("is_default_recovery"),
        sqlite_where=sa.text("is_default_recovery"),
    )
    op.drop_table("agentgraph")
