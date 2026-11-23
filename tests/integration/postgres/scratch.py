"""Helpers for the migration tests, which each migrate a scratch database."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, create_engine


@contextmanager
def connect(url: str) -> Iterator[Connection]:
    """A connection to ``url`` in a transaction committed on exit."""
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            yield connection
    finally:
        engine.dispose()
