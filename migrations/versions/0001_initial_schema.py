"""initial schema

The schema as of develop before the future robot contract, from an empty database.

Revision ID: 0001
Revises:
Create Date: 2026-11-16 19:25:14.532282

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "robotkey",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("key_hash", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "key_last4", sqlmodel.sql.sqltypes.AutoString(length=4), nullable=False
        ),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_robotkey_key_hash"), "robotkey", ["key_hash"], unique=False
    )
    op.create_table(
        "router",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("api_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("api_endpoint", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "provider_type",
            sa.Enum("OPENAI", "OPENROUTER", "VLLM", name="provider"),
            nullable=False,
        ),
        sa.Column(
            "capabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "vision_backbone",
            sa.Enum("OTHER", "QWEN2_5", "QWEN3_5", name="visionbackbone"),
            nullable=True,
        ),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("updated_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "tool",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("fn_module", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("updated_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fn_module"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "user",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("username", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("password", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("DEVELOPER", "ADMINISTRATOR", name="userrole"),
            nullable=False,
        ),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("updated_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_user_username"), "user", ["username"], unique=True)
    op.create_table(
        "agent",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("prompt", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("response_model", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column(
            "input_type", sa.Enum("TEXT", "IMAGETEXT", name="inputtype"), nullable=False
        ),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column(
            "reasoning_effort",
            sa.Enum(
                "MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH", name="reasoningeffort"
            ),
            nullable=True,
        ),
        sa.Column("router_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("updated_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "Agent", "GatewayAgent", "GuiAgent", "ErrorAgent", name="agenttype"
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["router_id"],
            ["router.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "robotexception",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "exception_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("robot_key_id", sa.Uuid(), nullable=True),
        sa.Column("infered_success", sa.Boolean(), nullable=False),
        sa.Column("reported_success", sa.Boolean(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("finished_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["robot_key_id"],
            ["robotkey.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "usersession",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("valid_until", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "agenttool",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.Column("tool_id", sa.Uuid(), nullable=False),
        sa.Column("limit", sa.Integer(), nullable=True),
        sa.Column("required", sa.Boolean(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
        ),
        sa.ForeignKeyConstraint(
            ["tool_id"],
            ["tool.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "agenttrace",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("robot_exception_id", sa.Uuid(), nullable=True),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.Column("inputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("messages", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("finished_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
        ),
        sa.ForeignKeyConstraint(
            ["robot_exception_id"],
            ["robotexception.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "argument",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("json_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("agent_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"],
            ["agent.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "recoverycontext",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("robot_exception_id", sa.Uuid(), nullable=True),
        sa.Column("task_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("platform", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("os", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("variables", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.ForeignKeyConstraint(
            ["robot_exception_id"],
            ["robotexception.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "subagent",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("parent_agent_id", sa.Uuid(), nullable=False),
        sa.Column("child_agent_id", sa.Uuid(), nullable=False),
        sa.Column("limit", sa.Integer(), nullable=True),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["child_agent_id"],
            ["agent.id"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_agent_id"],
            ["agent.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "guitrace",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_trace_id", sa.Uuid(), nullable=False),
        sa.Column("screenshot_key", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("action_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "action_content", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("finished_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_trace_id"], ["agenttrace.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "recoveryerroredactivity",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("recovery_context_id", sa.Uuid(), nullable=True),
        sa.Column("case_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("activity_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("event_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("activity_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("action_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("application", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("input", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("error_code", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "error_description", sqlmodel.sql.sqltypes.AutoString(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["recovery_context_id"],
            ["recoverycontext.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("recovery_context_id"),
    )
    op.create_table(
        "recoveryuilogentry",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("recovery_context_id", sa.Uuid(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("case_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("activity_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("event_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("activity_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("action_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("application", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("input", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column(
            "ui_element_target", sqlmodel.sql.sqltypes.AutoString(), nullable=True
        ),
        sa.Column("ui_group", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("timestamp", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("previous_state", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("current_state", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.ForeignKeyConstraint(
            ["recovery_context_id"],
            ["recoverycontext.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "subagenttrace",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("parent_trace_id", sa.Uuid(), nullable=False),
        sa.Column("child_trace_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["child_trace_id"],
            ["agenttrace.id"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_trace_id"],
            ["agenttrace.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "tooltrace",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_trace_id", sa.Uuid(), nullable=False),
        sa.Column("tool_id", sa.Uuid(), nullable=False),
        sa.Column("input", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.Column("finished_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["agent_trace_id"],
            ["agenttrace.id"],
        ),
        sa.ForeignKeyConstraint(
            ["tool_id"],
            ["tool.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("tooltrace")
    op.drop_table("subagenttrace")
    op.drop_table("recoveryuilogentry")
    op.drop_table("recoveryerroredactivity")
    op.drop_table("guitrace")
    op.drop_table("subagent")
    op.drop_table("recoverycontext")
    op.drop_table("argument")
    op.drop_table("agenttrace")
    op.drop_table("agenttool")
    op.drop_table("usersession")
    op.drop_table("robotexception")
    op.drop_table("agent")
    op.drop_index(op.f("ix_user_username"), table_name="user")
    op.drop_table("user")
    op.drop_table("tool")
    op.drop_table("router")
    op.drop_index(op.f("ix_robotkey_key_hash"), table_name="robotkey")
    op.drop_table("robotkey")
    for enum_name in (
        "agenttype",
        "inputtype",
        "provider",
        "reasoningeffort",
        "userrole",
        "visionbackbone",
    ):
        sa.Enum(name=enum_name).drop(op.get_bind(), checkfirst=True)
