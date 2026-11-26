"""Helpers the routers share."""

from collections.abc import Iterable
from typing import Protocol
from uuid import UUID

from fastapi import HTTPException, status
from sqlmodel import Session, SQLModel


class _Identified(Protocol):
    @property
    def id(self) -> UUID: ...


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
