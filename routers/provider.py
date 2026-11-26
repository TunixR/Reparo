from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from database.agents.models import Agent
from database.auth.models import User
from database.engine import commit_or_500
from database.general import SessionDep
from database.provider.models import (
    Router,
    RouterCreate,
    RouterPublic,
    RouterUpdate,
)
from middlewares.auth import get_current_user, require_admin
from routers.common import get_or_404

router = APIRouter(
    prefix="/provider", tags=["Provider"], dependencies=[Depends(get_current_user)]
)


def _validate_capabilities_against_agents(
    session: Session,
    router_obj: Router,
    capabilities: list[Router.Capability],
    router_name: str,
) -> None:
    """Reject capability changes that would break agents using this router."""
    agents = (
        session.exec(select(Agent).where(Agent.router_id == router_obj.id))
        .unique()
        .all()
    )
    for agent in agents:
        try:
            Agent.validate_capabilities_support_input_type(
                capabilities,
                agent.input_type,
                router_name=router_name,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            ) from e


def _save_changes(
    session: Session, router_obj: Router, changes: Mapping[str, object]
) -> RouterPublic:
    """Set the changed fields, stamp the update, commit and answer the router."""
    for field, value in changes.items():
        setattr(router_obj, field, value)
    router_obj.normalize_vision_backbone()
    router_obj.updated_at = datetime.now(UTC)

    session.add(router_obj)
    commit_or_500(session, "update router")
    session.refresh(router_obj)
    return RouterPublic.model_validate(router_obj)


@router.get(
    "/", response_model=Sequence[RouterPublic], summary="List all provider routers"
)
def list_routers(
    session: SessionDep,
) -> Sequence[RouterPublic]:
    routers = session.exec(select(Router)).all()
    return [RouterPublic.model_validate(r) for r in routers]


@router.get(
    "/{router_id}", response_model=RouterPublic, summary="Get provider router by ID"
)
def get_router(
    router_id: UUID,
    session: SessionDep,
    _current_user: Annotated[User, Depends(get_current_user)],
) -> RouterPublic:
    return RouterPublic.model_validate(get_or_404(session, Router, router_id))


@router.post(
    "/",
    response_model=RouterPublic,
    status_code=status.HTTP_201_CREATED,
    summary="Create provider router",
)
def create_router(
    payload: RouterCreate,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> RouterPublic:
    router_obj = Router(**payload.model_dump())
    session.add(router_obj)
    commit_or_500(session, "create router")
    session.refresh(router_obj)
    return RouterPublic.model_validate(router_obj)


@router.put(
    "/{router_id}",
    response_model=RouterPublic,
    summary="Replace provider router",
)
def replace_router(
    router_id: UUID,
    payload: RouterCreate,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> RouterPublic:
    router_obj = get_or_404(session, Router, router_id)
    _validate_capabilities_against_agents(
        session, router_obj, payload.capabilities, router_obj.model_name
    )
    return _save_changes(session, router_obj, payload.model_dump())


@router.patch(
    "/{router_id}",
    response_model=RouterPublic,
    summary="Update provider router",
)
def update_router(
    router_id: UUID,
    payload: RouterUpdate,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> RouterPublic:
    router_obj = get_or_404(session, Router, router_id)
    if payload.capabilities is not None:
        # The rejection names the router by the name it is getting.
        new_name = payload.model_name
        _validate_capabilities_against_agents(
            session,
            router_obj,
            payload.capabilities,
            router_obj.model_name if new_name is None else new_name,
        )
    return _save_changes(session, router_obj, payload.model_dump(exclude_none=True))


@router.delete(
    "/{router_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete provider router",
)
def delete_router(
    router_id: UUID,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> None:
    session.delete(get_or_404(session, Router, router_id))
    commit_or_500(session, "delete router")
