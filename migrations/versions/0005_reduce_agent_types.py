"""reduce agent types

Reduces ``agenttype`` to ``Agent`` and ``GuiAgent``: an agent's role in a recovery
comes from its place in the agent graph, so gateway and error agents become plain
agents. Postgres can't drop an enum value, so the enum is replaced.

The downgrade restores the four values, but which agents were gateways or error
agents is lost. As a best effort, the entry agent of the enabled default recovery
graph becomes a ``GatewayAgent`` again when the old gateway rule holds for it: it
backs only entry nodes of enabled graphs, and enters no enabled graph but the
default. ``ErrorAgent`` is not restored; no code branched on it.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-20 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_TYPES = ("Agent", "GatewayAgent", "GuiAgent", "ErrorAgent")
NEW_TYPES = ("Agent", "GuiAgent")


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "UPDATE agent SET type = 'Agent' WHERE type IN ('GatewayAgent', 'ErrorAgent')"
    )
    _replace_enum(NEW_TYPES)


def downgrade() -> None:
    """Downgrade schema."""
    _replace_enum(OLD_TYPES)
    op.execute(
        sa.text(
            """
            UPDATE agent SET type = 'GatewayAgent'
            WHERE id = (
                SELECT node.agent_id FROM agentgraphnode AS node
                JOIN agentgraph AS graph ON graph.id = node.agent_graph_id
                WHERE graph.is_default_recovery AND graph.enabled AND node.is_entry
            )
            AND type = 'Agent'
            AND NOT EXISTS (
                SELECT 1 FROM agentgraphnode AS node
                JOIN agentgraph AS graph ON graph.id = node.agent_graph_id
                WHERE node.agent_id = agent.id AND graph.enabled
                AND (NOT node.is_entry OR NOT graph.is_default_recovery)
            )
            """
        )
    )


def _replace_enum(values: Sequence[str]) -> None:
    labels = ", ".join(f"'{value}'" for value in values)
    op.execute(f"CREATE TYPE agenttype_new AS ENUM ({labels})")
    op.execute(
        "ALTER TABLE agent ALTER COLUMN type TYPE agenttype_new"
        " USING type::text::agenttype_new"
    )
    op.execute("DROP TYPE agenttype")
    op.execute("ALTER TYPE agenttype_new RENAME TO agenttype")
