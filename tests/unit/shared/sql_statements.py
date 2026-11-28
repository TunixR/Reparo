"""Record the SQL statements an engine sends, to count the round trips of a write."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, event


@contextmanager
def recorded_statements(engine: Engine) -> Iterator[list[str]]:
    """The statements ``engine`` sends inside the block, in order, upper-cased."""
    statements: list[str] = []

    def _record(*args: Any) -> None:
        statements.append(str(args[2]).strip().upper())

    event.listen(engine, "before_cursor_execute", _record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", _record)


def touching(statements: list[str], verb: str, table: str = "") -> list[str]:
    """The statements that start with ``verb`` and name ``table``, or any table."""
    return [s for s in statements if s.startswith(verb) and table.upper() in s]
