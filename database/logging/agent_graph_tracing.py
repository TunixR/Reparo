"""Writes the trace of an agent graph run while it runs, in sessions of its own.

Rows are inserted, then updated by id: a trace is never read back.
"""

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
from database.engine import open_session, update_row
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
        trace = AgentGraphTrace(
            agent_graph_id=agent_graph_id, robot_exception_id=robot_exception_id
        )
        # Read before the commit expires it, which would read the row back.
        trace_id = trace.id
        with open_session() as session:
            session.add(trace)
            session.commit()
        return cls(trace_id)

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
            if not update_row(
                session,
                AgentGraphTrace,
                self.id,
                status=status,
                failure_reason=failure_reason,
                terminal_node_key=terminal_node_key,
                answer=answer,
                finished_at=datetime.now(UTC),
            ):
                raise LookupError(f"The agent graph trace {self.id} is gone.")
            session.commit()

    @contextmanager
    def node(
        self, execution_index: int, node_key: str, entered_by: AgentGraphEdge | None
    ) -> Iterator[None]:
        """Open a node trace for the block, and close it with how the block ended.

        A disabled agent or a cancelled task closes it as cancelled, any other
        error as failed; the error propagates.
        """
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
        node_id = node_trace.id
        with open_session() as session:
            session.add(node_trace)
            session.commit()
        self._open_node_id = node_id
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
            self._close_node(node_id, status, error)

    def _close_node(
        self, node_id: UUID, status: AgentGraphNodeTraceStatus, error: str | None
    ) -> None:
        self._open_node_id = None
        with open_session() as session:
            if not update_row(
                session,
                AgentGraphNodeTrace,
                node_id,
                status=status,
                error=error,
                finished_at=datetime.now(UTC),
            ):
                raise LookupError(f"The agent graph node trace {node_id} is gone.")
            session.commit()

    def claim_open_node(self) -> UUID | None:
        """The open node trace, once: the node's own agent trace links to it."""
        if self._open_node_claimed:
            return None
        self._open_node_claimed = True
        return self._open_node_id
