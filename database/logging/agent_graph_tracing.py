"""Writes the trace of an agent graph run while it runs, in sessions of its own."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self
from uuid import UUID

from pydantic import BaseModel

from database.agent_graphs.conditions import CONDITION
from database.agent_graphs.runner import GraphOutcome, NodeCancelledError
from database.engine import open_session
from database.logging.models import (
    AgentGraphNodeTrace,
    AgentGraphNodeTraceStatus,
    AgentGraphTrace,
    AgentGraphTraceStatus,
)
from observability.events import brief
from observability.redaction import redact

if TYPE_CHECKING:
    from database.agent_graphs.models import AgentGraphEdge


def _describe_error(exc: BaseException) -> str:
    """The error's type and its bounded, redacted message."""
    message = redact(brief(exc))
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class AgentGraphRunTrace:
    """The agent graph trace of one run, and the node trace that is open, if any."""

    def __init__(self, trace_id: UUID) -> None:
        self.id = trace_id
        self._open_node_id: UUID | None = None
        self._open_node_claimed = False

    @classmethod
    def open(cls, agent_graph_id: UUID, robot_exception_id: UUID) -> Self:
        with open_session() as session:
            trace = AgentGraphTrace(
                agent_graph_id=agent_graph_id, robot_exception_id=robot_exception_id
            )
            session.add(trace)
            session.commit()
            return cls(trace.id)

    def close(self, outcome: GraphOutcome, answer: BaseModel) -> None:
        """Close the run as completed, or failed if it hit a limit, with the answer sent."""
        self._finish(
            AgentGraphTraceStatus.FAILED
            if outcome.failure_reason
            else AgentGraphTraceStatus.COMPLETED,
            failure_reason=outcome.failure_reason,
            terminal_node_key=outcome.terminal_node,
            answer=answer.model_dump(mode="json"),
        )

    def fail(self, exc: BaseException) -> None:
        """Close the run as failed with the error that ended it."""
        self._finish(AgentGraphTraceStatus.FAILED, failure_reason=_describe_error(exc))

    def _finish(
        self,
        status: AgentGraphTraceStatus,
        *,
        failure_reason: str | None,
        terminal_node_key: str | None = None,
        answer: dict[str, object] | None = None,
    ) -> None:
        with open_session() as session:
            trace = session.get_one(AgentGraphTrace, self.id)
            trace.status = status
            trace.failure_reason = failure_reason
            trace.terminal_node_key = terminal_node_key
            trace.answer = answer
            trace.finished_at = datetime.now(UTC)
            session.add(trace)
            session.commit()

    @contextmanager
    def node(
        self, execution_index: int, node_key: str, entered_by: AgentGraphEdge | None
    ) -> Iterator[None]:
        """Open a node trace for the block, and close it with how the block ended.

        A disabled agent or a cancelled task closes it as cancelled, any other
        error as failed; the error propagates.
        """
        with open_session() as session:
            node_trace = AgentGraphNodeTrace(
                agent_graph_trace_id=self.id,
                execution_index=execution_index,
                node_key=node_key,
                entry_edge_id=None if entered_by is None else entered_by.id,
                entry_condition=None
                if entered_by is None or entered_by.condition is None
                else CONDITION.dump_python(
                    CONDITION.validate_python(entered_by.condition),
                    mode="json",
                    by_alias=True,
                ),
            )
            session.add(node_trace)
            session.commit()
            self._open_node_id = node_trace.id
            self._open_node_claimed = False
        status, error = AgentGraphNodeTraceStatus.COMPLETED, None
        try:
            yield
        except (NodeCancelledError, asyncio.CancelledError) as exc:
            status, error = AgentGraphNodeTraceStatus.CANCELLED, _describe_error(exc)
            raise
        except Exception as exc:
            status, error = AgentGraphNodeTraceStatus.FAILED, _describe_error(exc)
            raise
        finally:
            self._close_node(status, error)

    def _close_node(self, status: AgentGraphNodeTraceStatus, error: str | None) -> None:
        node_id = self._open_node_id
        self._open_node_id = None
        with open_session() as session:
            node_trace = session.get_one(AgentGraphNodeTrace, node_id)
            node_trace.status = status
            node_trace.error = error
            node_trace.finished_at = datetime.now(UTC)
            session.add(node_trace)
            session.commit()

    def claim_open_node(self) -> UUID | None:
        """The open node trace, once: the node's own agent trace links to it."""
        if self._open_node_claimed:
            return None
        self._open_node_claimed = True
        return self._open_node_id
