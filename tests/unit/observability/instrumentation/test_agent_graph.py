"""How an agent graph run's ending becomes its outcome, span status and events."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from uuid import UUID

import pytest
from fastapi import WebSocketDisconnect
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import observability.instrumentation.agent_graph as agent_graph_module
from database.agent_graphs.runner import NodeCancelledError
from observability.instrumentation.agent_graph import agent_graph_run
from tests.unit.shared.log_events import logged_events

GRAPH_ID = UUID(int=3)
AGENT_ID = UUID(int=4)


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        agent_graph_module, "tracer", provider.get_tracer("tests.agent_graph")
    )
    yield exporter
    provider.shutdown()


def _span(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    [span] = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


@pytest.mark.parametrize(
    "ending",
    [asyncio.CancelledError(), WebSocketDisconnect()],
    ids=["task cancelled", "robot gone"],
)
def test_a_cancelled_run_is_no_failure(
    ending: BaseException,
    exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(type(ending)):
        with agent_graph_run(GRAPH_ID, "Graph") as run:
            with run.node(0, "gateway", AGENT_ID):
                pass
            with run.node(1, "gui", AGENT_ID):
                raise ending

    graph_run = _span(exporter, "agent_graph.run")
    assert graph_run.attributes is not None
    assert graph_run.attributes["agent_graph.outcome"] == "cancelled"
    assert graph_run.attributes["agent_graph.nodes_run"] == 1
    assert graph_run.attributes["agent_graph.terminal_node.key"] == "gateway"
    assert graph_run.status.status_code.name == "UNSET"
    assert all(
        not s.events
        for s in exporter.get_finished_spans()
        if s.name == "agent_graph.node"
    )
    assert logged_events(caplog, "agent_graph.failed") == []


def test_a_cancelled_node_fails_the_run_with_a_warning(
    exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(NodeCancelledError):
        with agent_graph_run(GRAPH_ID, "Graph") as run:
            with run.node(0, "gateway", AGENT_ID):
                raise NodeCancelledError("an agent was disabled")

    graph_run = _span(exporter, "agent_graph.run")
    assert graph_run.attributes is not None
    assert graph_run.attributes["agent_graph.outcome"] == "node_cancelled"
    assert graph_run.attributes["agent_graph.nodes_run"] == 0
    assert "agent_graph.terminal_node.key" not in graph_run.attributes
    assert graph_run.status.status_code.name == "ERROR"
    [failed] = logged_events(caplog, "agent_graph.failed")
    assert failed.levelno == logging.WARNING
    assert vars(failed)["error.type"] == "NodeCancelledError"
