"""index the trace foreign keys and the session user

Postgres doesn't index foreign keys by itself, so listing a recovery's traces, loading
a trace's children and deleting a recovery, a trace, an agent graph or one of its edges
scanned whole trace tables. This revision indexes the foreign keys those queries and
deletes go through, and a
recovery's agent traces by ``(robot_exception_id, created_at)`` for the time order
they are listed in. ``usersession.user_id`` is indexed for the sessions deleted with
their user.

The indexes are built with ``CREATE INDEX CONCURRENTLY``: a plain ``CREATE INDEX``
blocks writes to the table until it finishes, and recoveries write traces all the time.
A concurrent build can't run in a transaction, so this revision commits the migration
transaction first. A failed concurrent build leaves an invalid index of the same name
behind, which ``IF NOT EXISTS`` would keep; each index is dropped first, so running
the upgrade again rebuilds it.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-28 10:32:35.398119

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (index, table, columns), as the models declare them.
INDEXES: list[tuple[str, str, list[str]]] = [
    (
        "ix_agentgraphnodetrace_agent_trace_id",
        "agentgraphnodetrace",
        ["agent_trace_id"],
    ),
    (
        "ix_agentgraphnodetrace_entry_edge_id",
        "agentgraphnodetrace",
        ["entry_edge_id"],
    ),
    ("ix_agentgraphtrace_agent_graph_id", "agentgraphtrace", ["agent_graph_id"]),
    ("ix_agenttrace_agent_id", "agenttrace", ["agent_id"]),
    (
        "ix_agenttrace_robot_exception_id_created_at",
        "agenttrace",
        ["robot_exception_id", "created_at"],
    ),
    ("ix_guitrace_agent_trace_id", "guitrace", ["agent_trace_id"]),
    (
        "ix_recoverycontext_robot_exception_id",
        "recoverycontext",
        ["robot_exception_id"],
    ),
    (
        "ix_recoveryuilogentry_recovery_context_id",
        "recoveryuilogentry",
        ["recovery_context_id"],
    ),
    ("ix_robotexception_robot_key_id", "robotexception", ["robot_key_id"]),
    ("ix_subagenttrace_child_trace_id", "subagenttrace", ["child_trace_id"]),
    ("ix_subagenttrace_parent_trace_id", "subagenttrace", ["parent_trace_id"]),
    ("ix_tooltrace_agent_trace_id", "tooltrace", ["agent_trace_id"]),
    ("ix_tooltrace_tool_id", "tooltrace", ["tool_id"]),
    ("ix_usersession_user_id", "usersession", ["user_id"]),
]


def upgrade() -> None:
    """Upgrade schema."""
    with op.get_context().autocommit_block():
        for name, table, columns in INDEXES:
            op.drop_index(
                name, table_name=table, postgresql_concurrently=True, if_exists=True
            )
            op.create_index(name, table, columns, postgresql_concurrently=True)


def downgrade() -> None:
    """Downgrade schema."""
    with op.get_context().autocommit_block():
        for name, table, _ in reversed(INDEXES):
            op.drop_index(
                name, table_name=table, postgresql_concurrently=True, if_exists=True
            )
