"""An agent graph run is traced with inserts and by-id updates, never reading a trace back."""

from uuid import uuid4

import pytest
from pydantic import BaseModel
from sqlalchemy import Engine
from sqlmodel import Session

from database.agent_graphs.runner import GraphOutcome
from database.logging.agent_graph_tracing import AgentGraphRunTrace
from database.logging.models import (
    AgentGraphNodeTrace,
    AgentGraphNodeTraceStatus,
    AgentGraphTrace,
    AgentGraphTraceStatus,
)
from tests.unit.shared.sql_statements import recorded_statements


class _Answer(BaseModel):
    success: bool = True


def _selects(statements: list[str]) -> list[str]:
    return [s for s in statements if s.startswith("SELECT")]


def test_a_traced_run_writes_without_reading(app_engine: Engine):
    with recorded_statements(app_engine) as statements:
        run_trace = AgentGraphRunTrace.open(uuid4(), uuid4())
        with run_trace.node(0, "gateway", None):
            pass
        run_trace.close(GraphOutcome("gateway", None, 1), _Answer())

    assert _selects(statements) == []
    assert [s.split()[0] for s in statements] == [
        "INSERT",
        "INSERT",
        "UPDATE",
        "UPDATE",
    ]
    with Session(app_engine) as session:
        trace = session.get_one(AgentGraphTrace, run_trace.id)
        assert trace.status is AgentGraphTraceStatus.COMPLETED
        assert trace.terminal_node_key == "gateway"
        assert trace.answer == {"success": True}
        assert trace.finished_at is not None
        [node] = trace.node_traces
        assert node.status is AgentGraphNodeTraceStatus.COMPLETED
        assert node.finished_at is not None


def test_a_failed_node_and_run_are_traced_as_failed(app_engine: Engine):
    run_trace = AgentGraphRunTrace.open(uuid4(), uuid4())

    with pytest.raises(ValueError), run_trace.node(0, "gateway", None):
        raise ValueError("bad output")
    run_trace.fail(ValueError("bad output"))

    with Session(app_engine) as session:
        trace = session.get_one(AgentGraphTrace, run_trace.id)
        assert trace.status is AgentGraphTraceStatus.FAILED
        assert trace.failure_reason == "ValueError: bad output"
        [node] = trace.node_traces
        assert isinstance(node, AgentGraphNodeTrace)
        assert node.status is AgentGraphNodeTraceStatus.FAILED
        assert node.error == "ValueError: bad output"


def test_closing_a_trace_that_is_gone_raises(app_engine: Engine):  # pyright: ignore[reportUnusedParameter]
    run_trace = AgentGraphRunTrace(uuid4())

    with pytest.raises(LookupError):
        run_trace.fail(RuntimeError("boom"))
