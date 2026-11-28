from typing import Annotated
from uuid import UUID

from fastapi import HTTPException, status
from fastapi.params import Depends
from sqlalchemy import Engine, inspect
from sqlmodel import Session, SQLModel, create_engine, update

from settings import (
    POSTGRES_MAX_OVERFLOW,
    POSTGRES_POOL_RECYCLE,
    POSTGRES_POOL_SIZE,
    POSTGRES_POOL_TIMEOUT,
    POSTGRES_URL,
)

postgres_url = POSTGRES_URL


def create_general_engine(url: str) -> Engine:
    """The app's engine, with the configured pool.

    A pooled connection is pinged before use, so one dropped by a restarted
    Postgres is replaced instead of failing the request that draws it.
    """
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=POSTGRES_POOL_SIZE,
        max_overflow=POSTGRES_MAX_OVERFLOW,
        pool_timeout=POSTGRES_POOL_TIMEOUT,
        pool_recycle=POSTGRES_POOL_RECYCLE,
    )


general_engine = create_general_engine(postgres_url)


def get_session():
    with Session(general_engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def open_session(*, expire_on_commit: bool = True) -> Session:
    """A session of its own, outside any request, for writes that commit on their own.

    With ``expire_on_commit`` false, the objects it loaded stay readable after a
    commit, which ends its transaction and returns its connection to the pool.
    """
    return Session(general_engine, expire_on_commit=expire_on_commit)


def update_row(
    session: Session, model: type[SQLModel], row_id: UUID, **values: object
) -> bool:
    """Write ``values`` to the row of ``model`` with id ``row_id``, in one UPDATE.

    Nothing is read first, and no loaded object is synchronized. Returns whether
    the row exists.
    """
    statement = (
        update(model)
        .where(inspect(model).columns["id"] == row_id)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    return session.exec(statement).rowcount == 1


def commit_or_500(session: Session, action: str) -> None:
    """Commit, or roll back and raise a generic 500.

    The global HTTP exception handler logs the chained database error and
    records it on the request span, so callers need no logging of their own.
    """
    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to {action}.",
        ) from exc
