"""Telemetry for an agent graph run: the ``agent_graph.run`` span and a span per node.

The graph reports each node execution and how the run ended; this module turns that
into spans, span attributes and log events.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from fastapi import WebSocketDisconnect
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from database.agent_graphs.runner import NodeCancelledError
from observability import attributes as attrs
from observability.events import brief, error_type, log_event

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from opentelemetry.trace import Span

    from database.agent_graphs.runner import FailureReason

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def _is_cancellation(exc: BaseException) -> bool:
    """Whether ``exc`` ended the run because the robot aborted or went away."""
    return isinstance(exc, asyncio.CancelledError | WebSocketDisconnect)


@contextmanager
def agent_graph_run(
    agent_graph_id: UUID, name: str
) -> Iterator[AgentGraphRunTelemetry]:
    """Open the ``agent_graph.run`` span of one run; what escapes it ends the run."""
    with tracer.start_as_current_span(
        attrs.SPAN_AGENT_GRAPH_RUN,
        attributes={
            attrs.AGENT_GRAPH_ID: str(agent_graph_id),
            attrs.AGENT_GRAPH_NAME: brief(name),
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        telemetry = AgentGraphRunTelemetry(span, agent_graph_id)
        try:
            yield telemetry
        except (asyncio.CancelledError, Exception) as exc:
            telemetry.raised(exc)
            raise


class AgentGraphRunTelemetry:
    """The node executions of a run, and how it ended, on its span."""

    def __init__(self, span: Span, agent_graph_id: UUID) -> None:
        self._span = span
        self._agent_graph_id = agent_graph_id
        self._nodes_run = 0
        self._terminal_node: str | None = None

    @contextmanager
    def node(self, execution_index: int, key: str, agent_id: UUID) -> Iterator[None]:
        """The current ``agent_graph.node`` span while the node runs.

        An error that escapes the node marks it failed; the agent's own span
        records the exception.
        """
        with tracer.start_as_current_span(
            attrs.SPAN_AGENT_GRAPH_NODE,
            attributes={
                attrs.AGENT_GRAPH_NODE_KEY: key,
                attrs.AGENT_ID: str(agent_id),
                attrs.AGENT_GRAPH_NODE_EXECUTION_INDEX: execution_index,
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield
            except Exception as exc:
                if not _is_cancellation(exc):
                    span.set_attribute(attrs.ERROR_TYPE, type(exc).__name__)
                    span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
        self._nodes_run += 1
        self._terminal_node = key

    def completed(self, failure_reason: FailureReason | None) -> None:
        """The run ended on its own, or at the limit named by ``failure_reason``."""
        if failure_reason is None:
            self._finish("completed")
        else:
            self._failed(failure_reason)

    def raised(self, exc: BaseException) -> None:
        """The run ended with ``exc``: ``cancelled`` with the robot, else failed."""
        if _is_cancellation(exc):
            self._finish("cancelled")
            return
        outcome = "node_cancelled" if isinstance(exc, NodeCancelledError) else "error"
        self._failed(outcome, exc)
        self._span.set_status(Status(StatusCode.ERROR, outcome))

    def _failed(self, outcome: str, exc: BaseException | None = None) -> None:
        """Finish as ``outcome`` and log it; the turn owns an unexpected error."""
        self._finish(outcome)
        log_event(
            logger,
            attrs.EVENT_AGENT_GRAPH_FAILED,
            {
                attrs.AGENT_GRAPH_ID: self._agent_graph_id,
                attrs.AGENT_GRAPH_OUTCOME: outcome,
                attrs.AGENT_GRAPH_NODES_RUN: self._nodes_run,
                attrs.ERROR_TYPE: error_type(exc),
                attrs.EXCEPTION_MESSAGE: None if exc is None else brief(exc),
            },
            logging.WARNING,
        )

    def _finish(self, outcome: str) -> None:
        self._span.set_attributes(
            {
                attrs.AGENT_GRAPH_OUTCOME: outcome,
                attrs.AGENT_GRAPH_NODES_RUN: self._nodes_run,
            }
        )
        if self._terminal_node is not None:
            self._span.set_attribute(
                attrs.AGENT_GRAPH_TERMINAL_NODE_KEY, self._terminal_node
            )
