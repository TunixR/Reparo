from collections.abc import Sequence
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import select

from database.agent_graphs import service
from database.agents.behaviours import validate_agent
from database.agents.models import (
    Agent,
    AgentCreate,
    AgentUpdate,
    IgnoredField,
    SavedAgent,
)
from database.auth.models import User
from database.engine import commit_or_500
from database.general import SessionDep
from database.provider.models import Router
from middlewares.auth import get_current_user, require_admin

router = APIRouter(
    prefix="/agents",
    tags=["Agents"],
    dependencies=[Depends(get_current_user)],
)


@router.get("/", response_model=Sequence[Agent], summary="List all agents")
def list_agents(
    session: SessionDep,
) -> Sequence[Agent]:
    return session.exec(select(Agent)).unique().all()


@router.get("/{agent_id}", response_model=Agent, summary="Get agent by ID")
def get_agent(
    agent_id: UUID,
    session: SessionDep,
) -> Agent:
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )
    return agent


@router.post(
    "/",
    response_model=SavedAgent,
    status_code=status.HTTP_201_CREATED,
    summary="Create agent",
    description=(
        "Creates an agent. Its type's rules apply: a GUI agent (TooledGuiAgent, "
        "TextGuiAgent) gets the fixed GUI inputs and IMAGETEXT input whatever is "
        "sent, and only a TextGuiAgent takes, and needs, a parser_path. The "
        "response lists in ignored_fields what the type stores but doesn't use."
    ),
)
def create_agent(
    payload: AgentCreate,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> Any:
    router_obj = session.get(Router, payload.router_id)
    if not router_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Router not found",
        )

    try:
        agent = Agent(**payload.model_dump(), router=router_obj)
        ignored = validate_agent(agent)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e

    session.add(agent)
    commit_or_500(session, "create agent")
    session.refresh(agent)
    return _saved(agent, ignored)


@router.patch(
    "/{agent_id}",
    response_model=SavedAgent,
    summary="Update agent",
    description=(
        "Updates an agent; the rules of its resulting type apply as on create, and "
        "the response lists in ignored_fields what the type stores but doesn't use."
    ),
)
def update_agent(
    agent_id: UUID,
    payload: AgentUpdate,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> Any:
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    router_obj = session.get(Router, payload.router_id or agent.router_id)
    if not router_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Router not found",
        )

    try:
        ignored = service.update_agent(session, agent, payload, router_obj)
    except ValueError as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(e),
        ) from e
    except service.AgentGraphError as e:
        session.rollback()
        raise HTTPException(status_code=e.status_code, detail=e.detail()) from e

    session.add(agent)
    commit_or_500(session, "update agent")
    session.refresh(agent)
    return _saved(agent, ignored)


def _saved(agent: Agent, ignored: list[IgnoredField]) -> Any:
    """The response of create and update: the agent and the fields its type ignores."""
    return SavedAgent.model_validate(
        agent.model_dump() | {"ignored_fields": [f.model_dump() for f in ignored]}
    )


@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent",
)
def delete_agent(
    agent_id: UUID,
    session: SessionDep,
    _current_user: Annotated[User, Depends(require_admin)],
) -> None:
    agent = session.get(Agent, agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found",
        )

    try:
        service.delete_agent(session, agent)
    except service.AgentGraphError as e:
        session.rollback()
        raise HTTPException(status_code=e.status_code, detail=e.detail()) from e
    commit_or_500(session, "delete agent")
