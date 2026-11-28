import pytest
from fastapi import HTTPException
from sqlalchemy.pool import QueuePool
from sqlmodel import Session

import settings
from database.engine import commit_or_500, create_general_engine


class _FailingSession:
    def __init__(self) -> None:
        self.rolled_back = False

    def commit(self) -> None:
        raise RuntimeError("db password=hunter2")

    def rollback(self) -> None:
        self.rolled_back = True


def test_commit_or_500_rolls_back_and_hides_the_cause() -> None:
    session = _FailingSession()

    with pytest.raises(HTTPException) as excinfo:
        commit_or_500(session, "create agent")  # pyright: ignore[reportArgumentType]

    assert session.rolled_back
    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to create agent."
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_commit_or_500_commits(session: Session) -> None:
    commit_or_500(session, "noop")


def test_the_engine_pings_pooled_connections_and_pools_as_configured() -> None:
    # Built, not taken from the module: the client fixture swaps the module's engine.
    pool = create_general_engine(settings.POSTGRES_URL).pool

    assert isinstance(pool, QueuePool)
    assert pool._pre_ping is True  # pyright: ignore[reportPrivateUsage]
    assert pool.size() == settings.POSTGRES_POOL_SIZE
    assert pool._max_overflow == settings.POSTGRES_MAX_OVERFLOW  # pyright: ignore[reportPrivateUsage]
    assert pool.timeout() == settings.POSTGRES_POOL_TIMEOUT
    assert pool._recycle == settings.POSTGRES_POOL_RECYCLE  # pyright: ignore[reportPrivateUsage]
