"""The indexes the trace and session queries rely on stay declared on the models.

Postgres doesn't index foreign keys by itself: without these, listing a recovery's
traces, loading a trace's children and deleting a recovery, a trace or a graph edge
scan whole trace tables. Revision 0009 builds the same indexes.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlmodel import SQLModel

import database.general  # noqa: F401 - registers every table on SQLModel.metadata

REVISION = (
    Path(__file__).resolve().parents[3]
    / "migrations"
    / "versions"
    / "0009_index_trace_foreign_keys.py"
)

EXPECTED_INDEXES = [
    # Filters a recovery's agent traces and sorts them by time (``_traces_of_recovery``).
    ("agenttrace", ("robot_exception_id", "created_at")),
    # Every Agent load joins its traces (``Agent.traces``).
    ("agenttrace", ("agent_id",)),
    # A trace's GUI and tool steps; GUI and tool trace lists join through them.
    ("guitrace", ("agent_trace_id",)),
    ("tooltrace", ("agent_trace_id",)),
    # Every Tool load joins its traces (``Tool.traces``).
    ("tooltrace", ("tool_id",)),
    # A trace's sub-agent links, and the sub-agent trace checks on insert.
    ("subagenttrace", ("parent_trace_id",)),
    ("subagenttrace", ("child_trace_id",)),
    # The node and graph traces an agent trace, edge or graph delete sets to null.
    ("agentgraphnodetrace", ("agent_trace_id",)),
    ("agentgraphnodetrace", ("entry_edge_id",)),
    ("agentgraphtrace", ("agent_graph_id",)),
    # A recovery's context and its UI log.
    ("recoverycontext", ("robot_exception_id",)),
    ("recoveryuilogentry", ("recovery_context_id",)),
    # The recoveries of a robot key.
    ("robotexception", ("robot_key_id",)),
    # A user's sessions, deleted with the user.
    ("usersession", ("user_id",)),
]


@pytest.mark.parametrize(("table", "columns"), EXPECTED_INDEXES)
def test_table_has_index(table: str, columns: tuple[str, ...]) -> None:
    indexed = {
        tuple(column.name for column in index.columns)
        for index in SQLModel.metadata.tables[table].indexes
    }

    assert columns in indexed


def _load_revision() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_0009", REVISION)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "table", "columns"), _load_revision().INDEXES)
def test_revision_builds_the_index_the_model_declares(
    name: str, table: str, columns: list[str]
) -> None:
    declared = {
        str(index.name): [column.name for column in index.columns]
        for index in SQLModel.metadata.tables[table].indexes
    }

    assert declared.get(name) == columns
