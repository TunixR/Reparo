"""Migration 0007 adds the router's context window and backfills existing routers.

Each test migrates a scratch database of its own on the configured server, so the
framework's database is never touched. The scratch database is dropped afterwards.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


def _insert_router(url: str) -> None:
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            _ = connection.execute(
                text(
                    "INSERT INTO router (id, api_key, model_name, api_endpoint, "
                    "provider_type, capabilities, created_at, updated_at) VALUES "
                    "(:id, '', 'm', 'http://x', 'OPENAI', '{}'::jsonb, :now, :now)"
                ),
                {"id": uuid.uuid4(), "now": datetime.now(UTC)},
            )
    finally:
        engine.dispose()


def _context_window_limits(url: str) -> list[int | None]:
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return list(
                connection.execute(
                    text("SELECT context_window_limit FROM router")
                ).scalars()
            )
    finally:
        engine.dispose()


def test_upgrade_backfills_existing_routers_with_250k(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0006")
    _insert_router(scratch_url)

    command.upgrade(alembic_config, "0007")

    assert _context_window_limits(scratch_url) == [250_000]


def test_the_column_stays_nullable_without_a_server_default(
    scratch_url: str, alembic_config: Config
) -> None:
    command.upgrade(alembic_config, "0007")
    _insert_router(scratch_url)

    # The application sets the default; the column only backfilled existing rows.
    assert _context_window_limits(scratch_url) == [None]
