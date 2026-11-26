from __future__ import annotations

import secrets
from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlmodel import Session, select

from database.auth.models import User
from database.engine import commit_or_500
from database.general import SessionDep
from database.keys.models import (
    RobotKey,
    RobotKeyCreate,
    RobotKeyCreated,
    RobotKeyPublic,
)
from middlewares.auth import get_current_user, require_admin
from routers.common import get_or_404
from security.utils import robot_key_hash

router = APIRouter(
    prefix="/keys", tags=["Keys"], dependencies=[Depends(get_current_user)]
)


def _key(session: Session, key_id: UUID) -> RobotKey:
    return get_or_404(session, RobotKey, key_id, "Key not found")


def _public(key: RobotKey) -> RobotKeyPublic:
    """The key as listed: its plaintext is never shown again, only its last 4."""
    return RobotKeyPublic(
        id=key.id,
        name=key.name,
        description=key.description,
        enabled=key.enabled,
        key=f"****{key.key_last4}",
        created_at=key.created_at,
    )


@router.get("", response_model=Sequence[RobotKeyPublic], summary="List RobotKeys")
def list_robot_keys(
    session: SessionDep,
) -> Sequence[RobotKeyPublic]:
    keys = session.exec(select(RobotKey)).all()
    return [_public(k) for k in keys]


@router.get("/{key_id}", response_model=RobotKeyPublic, summary="Get RobotKey by ID")
def get_robot_key(
    key_id: UUID,
    session: SessionDep,
) -> RobotKeyPublic:
    return _public(_key(session, key_id))


@router.post(
    "",
    response_model=RobotKeyCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Create RobotKey",
)
def create_robot_key(
    payload: RobotKeyCreate,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> RobotKeyCreated:
    plaintext_key = secrets.token_urlsafe(24)
    key = RobotKey(
        name=payload.name,
        description=payload.description,
        enabled=payload.enabled,
        key_hash=robot_key_hash(plaintext_key),
        key_last4=plaintext_key[-4:],
    )

    session.add(key)
    commit_or_500(session, "create key")
    session.refresh(key)

    return RobotKeyCreated(
        id=key.id,
        name=key.name,
        description=key.description,
        enabled=key.enabled,
        key=plaintext_key,
        created_at=key.created_at,
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete RobotKey",
)
def delete_robot_key(
    key_id: UUID,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> None:
    session.delete(_key(session, key_id))
    commit_or_500(session, "delete key")


@router.post(
    "/toggle/{key_id}",
    response_model=RobotKeyPublic,
    summary="Toggle RobotKey enabled",
)
def toggle_robot_key(
    key_id: UUID,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> RobotKeyPublic:
    key = _key(session, key_id)

    key.enabled = not key.enabled
    session.add(key)
    commit_or_500(session, "toggle key")
    session.refresh(key)

    return _public(key)
