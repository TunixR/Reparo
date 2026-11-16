"""future contract columns

Adds the columns of the future robot contract: the robot's capabilities on the recovery
context, the process model link (model_event_id) on UI log entries and errored activities,
and the robot's abort reason on the exception.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-16 19:25:32.224080

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# RecoveryCapabilities().model_dump(), written to existing recovery contexts.
DEFAULT_CAPABILITIES = '{"modalities": ["vision"], "ocr": false, "details": {}}'


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "recoverycontext",
        sa.Column(
            "capabilities",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text(f"'{DEFAULT_CAPABILITIES}'::jsonb"),
        ),
    )
    # The default only backfills existing rows; the application sets the value.
    op.alter_column("recoverycontext", "capabilities", server_default=None)
    op.add_column(
        "recoveryerroredactivity",
        sa.Column("model_event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "recoveryuilogentry",
        sa.Column("model_event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "robotexception",
        sa.Column("abort_code", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "robotexception",
        sa.Column(
            "abort_description", sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("robotexception", "abort_description")
    op.drop_column("robotexception", "abort_code")
    op.drop_column("recoveryuilogentry", "model_event_id")
    op.drop_column("recoveryerroredactivity", "model_event_id")
    op.drop_column("recoverycontext", "capabilities")
