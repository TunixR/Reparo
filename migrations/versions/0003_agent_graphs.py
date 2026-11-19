"""agent graphs

Adds the agent graph tables: graphs, their nodes (each backed by an agent) and the
edges between nodes. New tables only, so there is nothing to backfill.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-19 10:15:09.179265

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "agentgraph",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("max_node_executions", sa.Integer(), nullable=True),
        sa.Column("execution_timeout_s", sa.Float(), nullable=True),
        sa.Column("node_timeout_s", sa.Float(), nullable=True),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("updated_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
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


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_agentgraphedge_agent_graph_id"), table_name="agentgraphedge")
    op.drop_table("agentgraphedge")
    op.drop_index(op.f("ix_agentgraphnode_agent_graph_id"), table_name="agentgraphnode")
    op.drop_table("agentgraphnode")
    op.drop_table("agentgraph")
