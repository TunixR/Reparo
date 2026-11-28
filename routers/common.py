"""Helpers the routers share."""

from collections.abc import Iterable
from datetime import datetime
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlmodel import Session, SQLModel, col
from sqlmodel.sql.expression import SelectOfScalar


class _Identified(Protocol):
    @property
    def id(self) -> UUID: ...


class _Dated(Protocol):
    id: UUID
    created_at: datetime


class Pagination(BaseModel):
    """Which page of a list to answer: ``limit`` rows after the first ``offset``."""

    limit: int = Field(default=100, ge=1, le=1000, description="Rows per page.")
    offset: int = Field(default=0, ge=0, description="Rows to skip.")

    def newest_first[T: _Dated](
        self, query: SelectOfScalar[T], row: type[T]
    ) -> SelectOfScalar[T]:
        """The page of ``query``, newest ``row`` first.

        Rows created at the same time are ordered by id, so the pages of a list
        don't overlap or skip a row.
        """
        return (
            query.order_by(col(row.created_at).desc(), col(row.id).desc())
            .offset(self.offset)
            .limit(self.limit)
        )


PageQuery = Annotated[Pagination, Query()]


def not_found(detail: str) -> HTTPException:
    """The 404 a router answers when a row it was asked for doesn't exist."""
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def get_or_404[T: SQLModel](
    session: Session, model: type[T], row_id: UUID, detail: str | None = None
) -> T:
    """The ``model`` row with ``row_id``, or a 404 (by default ``<Model> not found``)."""
    row = session.get(model, row_id)
    if row is None:
        raise not_found(detail or f"{model.__name__} not found")
    return row


def find_or_404[T: _Identified](rows: Iterable[T], row_id: UUID, detail: str) -> T:
    """The row of ``rows`` with ``row_id``, or a 404 with ``detail``."""
    row = next((r for r in rows if r.id == row_id), None)
    if row is None:
        raise not_found(detail)
    return row
