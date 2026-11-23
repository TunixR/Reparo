"""add the router's context window limit

``router.context_window_limit`` is the model's context window in tokens, passed to
Strands as the ``context_window_limit`` model config. It is nullable and starts unset,
which keeps today's behaviour: Strands looks the model id up in its bundled table.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-23 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "router", sa.Column("context_window_limit", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("router", "context_window_limit")
