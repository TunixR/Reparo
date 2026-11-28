from collections.abc import Sequence
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import select

from database.general import SessionDep
from database.tools.models import Tool
from middlewares.auth import get_current_user
from routers.common import get_or_404

router = APIRouter(
    prefix="/tools", tags=["Tools"], dependencies=[Depends(get_current_user)]
)


@router.get("/", response_model=Sequence[Tool], summary="List all tools")
def list_tools(
    session: SessionDep,
) -> Sequence[Tool]:
    return session.exec(select(Tool)).all()


@router.get("/{tool_id}", response_model=Tool, summary="Get tool by ID")
def get_tool(
    tool_id: UUID,
    session: SessionDep,
) -> Tool:
    return get_or_404(session, Tool, tool_id)
