import zipfile
from collections.abc import Sequence
from io import BytesIO
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, col, select
from sqlmodel.sql.expression import SelectOfScalar

from database.engine import commit_or_500
from database.general import SessionDep
from database.logging.models import (
    AgentGraphTrace,
    AgentGraphTraceRead,
    AgentTrace,
    GUITrace,
    RecoveryContext,
    RobotException,
    ToolTrace,
)
from middlewares.auth import get_current_user
from routers.common import PageQuery, Pagination, get_or_404, not_found
from s3.utils import S3Client


class RecoveryContextEnriched(BaseModel):
    id: UUID
    robot_exception_id: UUID
    task_name: str
    platform: str
    os: str
    variables: dict[str, Any]
    model: str
    ui_log: list[dict[str, Any]]
    errored_act: dict[str, Any] | None


router = APIRouter(
    prefix="/logging",
    tags=["Logging"],
    dependencies=[Depends(get_current_user)],
)


class RecoveryPage(Pagination):
    """A page of the history of every recovery, or of one."""

    robot_exception_id: UUID | None = Field(
        default=None, description="Only the traces of this robot exception."
    )


RecoveryPageQuery = Annotated[RecoveryPage, Query()]


def _traces_of_recovery[T: (AgentTrace, GUITrace, ToolTrace)](
    trace: type[T], robot_exception_id: UUID | None
) -> SelectOfScalar[T]:
    """Traces, only those of one recovery if given.

    GUI and tool traces belong to a recovery through their agent trace.
    """
    query = select(trace)
    if robot_exception_id is None:
        return query
    if trace is not AgentTrace:
        query = query.join(AgentTrace)
    return query.where(AgentTrace.robot_exception_id == robot_exception_id)


def _get_recovery_context_or_404(
    *,
    exception_id: UUID,
    session: SessionDep,
) -> RecoveryContext:
    _ = get_or_404(session, RobotException, exception_id)
    recovery_context = session.exec(
        select(RecoveryContext).where(
            RecoveryContext.robot_exception_id == exception_id
        )
    ).first()
    if not recovery_context:
        raise not_found("RecoveryContext not found")
    return recovery_context


def _list_traces[T: (AgentTrace, GUITrace, ToolTrace)](
    session: Session, trace: type[T], page: RecoveryPage
) -> Sequence[T]:
    """A page of traces, newest first; only those of one recovery if given."""
    query = page.newest_first(
        _traces_of_recovery(trace, page.robot_exception_id), trace
    )
    return session.exec(query).all()


def _delete(session: Session, row: SQLModel) -> None:
    name = type(row).__name__
    session.delete(row)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The {name} is still referenced by other records.",
        ) from exc
    commit_or_500(session, f"delete {name}")


@router.get(
    "/markdown/",
    description="Retrieve the markdown log for a given agent trace ID.",
    summary="Get Agent Trace Markdown Log",
    responses={
        404: {"description": "AgentTrace not found"},
        200: {"content": {"text/markdown": {}}},
    },
)
async def get_agent_trace_markdown(agent_trace_id: UUID, session: SessionDep):
    """
    Endpoint to retrieve the markdown log for a given agent trace ID.
    """
    agent_trace = get_or_404(
        session,
        AgentTrace,
        agent_trace_id,
        f"AgentTrace with ID {agent_trace_id} not found.",
    )
    markdown_log = await agent_trace.get_makdown_log()
    return Response(content=markdown_log, media_type="text/markdown")


_UI_LOG_HEADER = "timestamp,event_type,click_x,click_y,text,screenshot\n"


def _click(box: Sequence[Any], default: tuple[str, str]) -> tuple[str, str]:
    """A box's relative ([0,1]) click coordinates, or ``default`` if there is none."""
    return (str(box[0]), str(box[1])) if box else default


def _ui_log_rows(gui_trace: GUITrace) -> list[str]:
    """The UI log CSV rows of one GUI action: none, one, or a start and an end."""
    event_type = gui_trace.action_type
    kind = event_type.lower()
    if kind in ("wait", "finish"):
        return []
    content = gui_trace.action_content
    timestamp = gui_trace.finished_at.isoformat() if gui_trace.finished_at else ""
    text = content.get("content", "")
    # A step without a screenshot keeps its row, with the screenshot cell empty.
    key = gui_trace.screenshot_key
    screenshot = f"{key}.jpeg" if key is not None else ""

    def row(event: str, click: tuple[str, str]) -> str:
        return f"{timestamp},{event},{click[0]},{click[1]},{text},{screenshot}\n"

    start = _click(content.get("start_box", []), ("", ""))
    if kind == "scroll":
        return [row(f"scroll_{content.get('direction', '')}", start)]
    if kind in ("drag", "select"):
        end = _click(content.get("end_box", []), start)
        return [row(f"{event_type}_start", start), row(f"{event_type}_end", end)]
    return [row(event_type, start)]


def _ui_log_zip(ui_log: str, screenshots: dict[str, bytes]) -> bytes:
    """A zip of the UI log CSV and a ``screenshots`` folder of its images."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("ui_log.csv", data=ui_log)
        zipf.mkdir("screenshots")
        for key, img_bytes in screenshots.items():
            zipf.writestr(f"screenshots/{key}.jpeg", data=img_bytes)
    return buffer.getvalue()


@router.get(
    "/ui_log/",
    description="Retrieve the UI log for a given robot exception ID, alongside screenshots",
    summary="Get Robot Exception UI Log",
    responses={
        404: {"description": "RobotException not found"},
        200: {"content": {"text/csv": {}}},
    },
)
async def get_exception_ui_log(exception_id: UUID, session: SessionDep):
    """
    Retrieve the UI log for a given robot exception ID.
    """
    _ = get_or_404(
        session,
        RobotException,
        exception_id,
        f"RobotException with ID {exception_id} not found.",
    )
    gui_traces = session.exec(
        _traces_of_recovery(GUITrace, exception_id)
        .where(col(GUITrace.success))
        .order_by(col(GUITrace.created_at))
    ).all()
    ui_log = _UI_LOG_HEADER + "".join(
        row for gui_trace in gui_traces for row in _ui_log_rows(gui_trace)
    )
    screenshots = await S3Client.bulk_download_bytes(
        [
            key
            for gui_trace in gui_traces
            if (key := gui_trace.screenshot_key) is not None
        ]
    )
    return StreamingResponse(
        iter([_ui_log_zip(ui_log, screenshots)]), media_type="application/zip"
    )


@router.get(
    "/agent_traces", response_model=Sequence[AgentTrace], summary="List AgentTraces"
)
def list_agent_traces(
    session: SessionDep, page: RecoveryPageQuery
) -> Sequence[AgentTrace]:
    return _list_traces(session, AgentTrace, page)


@router.get(
    "/agent_traces/{trace_id}",
    response_model=AgentTrace,
    summary="Get AgentTrace by ID",
)
def get_agent_trace(
    trace_id: UUID,
    session: SessionDep,
) -> AgentTrace:
    return get_or_404(session, AgentTrace, trace_id)


@router.delete(
    "/agent_traces/{trace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete AgentTrace",
)
def delete_agent_trace(
    trace_id: UUID,
    session: SessionDep,
) -> None:
    _delete(session, get_or_404(session, AgentTrace, trace_id))


@router.get(
    "/agent_graph_traces",
    response_model=Sequence[AgentGraphTraceRead],
    summary="List AgentGraphTraces",
)
def list_agent_graph_traces(
    session: SessionDep, page: RecoveryPageQuery
) -> Sequence[AgentGraphTrace]:
    """A page of agent graph runs, newest first; only those of one recovery if given."""
    query = select(AgentGraphTrace)
    if page.robot_exception_id is not None:
        query = query.where(
            AgentGraphTrace.robot_exception_id == page.robot_exception_id
        )
    return session.exec(page.newest_first(query, AgentGraphTrace)).all()


def _get_agent_graph_trace_or_404(
    trace_id: UUID, session: SessionDep
) -> AgentGraphTrace:
    return get_or_404(session, AgentGraphTrace, trace_id)


@router.get(
    "/agent_graph_traces/{trace_id}",
    response_model=AgentGraphTraceRead,
    summary="Get AgentGraphTrace by ID",
)
def get_agent_graph_trace(trace_id: UUID, session: SessionDep) -> AgentGraphTrace:
    return _get_agent_graph_trace_or_404(trace_id, session)


@router.get(
    "/agent_graph_traces/{trace_id}/markdown",
    summary="Get AgentGraphTrace Markdown Log",
    description="The run's header, then each node execution with its agent trace.",
    responses={
        404: {"description": "AgentGraphTrace not found"},
        200: {"content": {"text/markdown": {}}},
    },
)
async def get_agent_graph_trace_markdown(
    trace_id: UUID, session: SessionDep
) -> Response:
    trace = _get_agent_graph_trace_or_404(trace_id, session)
    return Response(content=await trace.get_markdown_log(), media_type="text/markdown")


@router.get("/gui_traces", response_model=Sequence[GUITrace], summary="List GUITraces")
def list_gui_traces(session: SessionDep, page: RecoveryPageQuery) -> Sequence[GUITrace]:
    return _list_traces(session, GUITrace, page)


@router.get(
    "/gui_traces/{gui_trace_id}", response_model=GUITrace, summary="Get GUITrace by ID"
)
def get_gui_trace(
    gui_trace_id: UUID,
    session: SessionDep,
) -> GUITrace:
    return get_or_404(session, GUITrace, gui_trace_id)


@router.delete(
    "/gui_traces/{gui_trace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete GUITrace",
)
def delete_gui_trace(
    gui_trace_id: UUID,
    session: SessionDep,
) -> None:
    _delete(session, get_or_404(session, GUITrace, gui_trace_id))


@router.get(
    "/tool_traces", response_model=Sequence[ToolTrace], summary="List ToolTraces"
)
def list_tool_traces(
    session: SessionDep, page: RecoveryPageQuery
) -> Sequence[ToolTrace]:
    return _list_traces(session, ToolTrace, page)


@router.get(
    "/tool_traces/{tool_trace_id}",
    response_model=ToolTrace,
    summary="Get ToolTrace by ID",
)
def get_tool_trace(
    tool_trace_id: UUID,
    session: SessionDep,
) -> ToolTrace:
    return get_or_404(session, ToolTrace, tool_trace_id)


@router.delete(
    "/tool_traces/{tool_trace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete ToolTrace",
)
def delete_tool_trace(
    tool_trace_id: UUID,
    session: SessionDep,
) -> None:
    _delete(session, get_or_404(session, ToolTrace, tool_trace_id))


@router.get(
    "/robot_exceptions",
    response_model=Sequence[RobotException],
    summary="List RobotExceptions",
)
def list_robot_exceptions(
    session: SessionDep, page: PageQuery
) -> Sequence[RobotException]:
    """A page of robot exceptions, newest first."""
    return session.exec(page.newest_first(select(RobotException), RobotException)).all()


@router.get(
    "/key/{key_id}",
    response_model=Sequence[RobotException],
    summary="List RobotExceptions by RobotKey",
)
def list_robot_exceptions_by_key(
    key_id: UUID,
    session: SessionDep,
    page: PageQuery,
) -> Sequence[RobotException]:
    """A page of the robot exceptions submitted with one key, newest first."""
    query = select(RobotException).where(RobotException.robot_key_id == key_id)
    return session.exec(page.newest_first(query, RobotException)).all()


@router.get(
    "/robot_exceptions/{exception_id}",
    response_model=RobotException,
    summary="Get RobotException by ID",
)
def get_robot_exception(
    exception_id: UUID,
    session: SessionDep,
) -> RobotException:
    return get_or_404(session, RobotException, exception_id)


@router.delete(
    "/robot_exceptions/{exception_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete RobotException",
)
def delete_robot_exception(
    exception_id: UUID,
    session: SessionDep,
) -> None:
    _delete(session, get_or_404(session, RobotException, exception_id))


@router.get(
    "/recovery_context/{exception_id}",
    response_model=RecoveryContextEnriched,
    summary="Get normalized recovery context by exception ID",
)
def get_recovery_context(
    exception_id: UUID,
    session: SessionDep,
) -> RecoveryContextEnriched:
    recovery_context = _get_recovery_context_or_404(
        exception_id=exception_id,
        session=session,
    )
    return RecoveryContextEnriched.model_validate(
        recovery_context.model_dump()
        | {
            "id": recovery_context.id,
            "robot_exception_id": recovery_context.robot_exception_id,
        }
    )
