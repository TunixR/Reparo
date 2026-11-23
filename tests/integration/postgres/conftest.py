"""Fixtures for the Postgres integration tests.

The suite needs the Postgres configured in ``settings.py`` / ``.env`` (in
development, ``docker/compose.dev.yaml``) and is skipped when it is not
reachable. Tests never commit: each one runs in a transaction that is rolled
back, so the database is left as it was. Migration tests instead migrate a
scratch database (``scratch_url``) that is dropped afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

import settings
from database.general import ALEMBIC_INI


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    engine = create_engine(settings.POSTGRES_URL, connect_args={"connect_timeout": 3})
    try:
        with engine.connect():
            pass
    except OperationalError as exc:
        engine.dispose()
        pytest.skip(
            f"Postgres not reachable at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}: {exc}"
        )
    yield engine
    engine.dispose()


@pytest.fixture
def pg_session(pg_engine: Engine) -> Iterator[Session]:
    """A session whose work is rolled back at the end of the test."""
    with pg_engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection) as session:
            yield session
        transaction.rollback()


@pytest.fixture
def scratch_url(pg_engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """The URL of an empty database that ``alembic`` migrates instead of the real one."""
    name = f"reparo_migration_{uuid.uuid4().hex[:12]}"
    admin = pg_engine.execution_options(isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        _ = connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(settings.POSTGRES_URL).set(database=name)
    rendered = url.render_as_string(hide_password=False)
    # migrations/env.py reads the URL from settings each time a command runs.
    monkeypatch.setattr(settings, "POSTGRES_URL", rendered)
    try:
        yield rendered
    finally:
        with admin.connect() as connection:
            _ = connection.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture
def alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))
