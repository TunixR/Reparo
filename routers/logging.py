import asyncio
import zipfile
from collections.abc import Sequence
from io import BytesIO
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import raiseload
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
from routers.common import get_or_404, not_found
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


RobotExceptionFilter = Annotated[
    UUID | None,
    Query(description="Only the traces of this robot exception."),
]


def _traces_of_recovery[T: (AgentTrace, GUITrace, ToolTrace)](
    trace: type[T], robot_exception_id: UUID | None, *, oldest_first: bool = False
) -> SelectOfScalar[T]:
    """Traces newest first; only those of one recovery if given.

    GUI and tool traces belong to a recovery through their agent trace.
    """
    created_at = col(trace.created_at)
    query = select(trace).order_by(created_at if oldest_first else created_at.desc())
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
    session: Session, trace: type[T], robot_exception_id: UUID | None
) -> Sequence[T]:
    return session.exec(_traces_of_recovery(trace, robot_exception_id)).unique().all()


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


# The export endpoints are async for their S3 downloads. Their database reads are `def`
# dependencies, which FastAPI runs in its threadpool. They load everything the export
# reads, so the async part runs no SQL on the event loop.


def _agent_trace_tree_or_404(agent_trace_id: UUID, session: SessionDep) -> AgentTrace:
    agent_trace = AgentTrace.get_with_markdown_tree(session, agent_trace_id)
    if agent_trace is None:
        raise not_found(f"AgentTrace with ID {agent_trace_id} not found.")
    return agent_trace


AgentTraceTreeDep = Annotated[AgentTrace, Depends(_agent_trace_tree_or_404)]


@router.get(
    "/markdown/",
    description="Retrieve the markdown log for a given agent trace ID.",
    summary="Get Agent Trace Markdown Log",
    responses={
        404: {"description": "AgentTrace not found"},
        200: {"content": {"text/markdown": {}}},
    },
)
async def get_agent_trace_markdown(agent_trace: AgentTraceTreeDep) -> Response:
    """
    Endpoint to retrieve the markdown log for a given agent trace ID.
    """
    markdown_log = await agent_trace.get_markdown_log()
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


def _successful_gui_steps_or_404(
    exception_id: UUID, session: SessionDep
) -> Sequence[GUITrace]:
    """The recovery's successful GUI steps in the order they ran, without relationships."""
    _ = get_or_404(
        session,
        RobotException,
        exception_id,
        f"RobotException with ID {exception_id} not found.",
    )
    return session.exec(
        _traces_of_recovery(GUITrace, exception_id, oldest_first=True)
        .where(col(GUITrace.success))
        .options(raiseload("*"))
    ).all()


SuccessfulGUIStepsDep = Annotated[
    Sequence[GUITrace], Depends(_successful_gui_steps_or_404)
]


@router.get(
    "/ui_log/",
    description="Retrieve the UI log for a given robot exception ID, alongside screenshots",
    summary="Get Robot Exception UI Log",
    responses={
        404: {"description": "RobotException not found"},
        200: {"content": {"text/csv": {}}},
    },
)
async def get_exception_ui_log(
    gui_traces: SuccessfulGUIStepsDep,
) -> StreamingResponse:
    """
    Retrieve the UI log for a given robot exception ID.
    """
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
    archive = await asyncio.to_thread(_ui_log_zip, ui_log, screenshots)
    return StreamingResponse(iter([archive]), media_type="application/zip")


@router.get(
    "/agent_traces", response_model=Sequence[AgentTrace], summary="List AgentTraces"
)
def list_agent_traces(
    session: SessionDep, robot_exception_id: RobotExceptionFilter = None
) -> Sequence[AgentTrace]:
    return _list_traces(session, AgentTrace, robot_exception_id)


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
    session: SessionDep, robot_exception_id: RobotExceptionFilter = None
) -> Sequence[AgentGraphTrace]:
    """Agent graph runs, newest first; only those of one recovery if given."""
    query = select(AgentGraphTrace).order_by(col(AgentGraphTrace.created_at).desc())
    if robot_exception_id is not None:
        query = query.where(AgentGraphTrace.robot_exception_id == robot_exception_id)
    return session.exec(query).all()


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


def _agent_graph_trace_tree_or_404(
    trace_id: UUID, session: SessionDep
) -> AgentGraphTrace:
    trace = AgentGraphTrace.get_with_markdown_tree(session, trace_id)
    if trace is None:
        raise not_found("AgentGraphTrace not found")
    return trace


AgentGraphTraceTreeDep = Annotated[
    AgentGraphTrace, Depends(_agent_graph_trace_tree_or_404)
]


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
    trace: AgentGraphTraceTreeDep,
) -> Response:
    return Response(content=await trace.get_markdown_log(), media_type="text/markdown")


@router.get("/gui_traces", response_model=Sequence[GUITrace], summary="List GUITraces")
def list_gui_traces(
    session: SessionDep, robot_exception_id: RobotExceptionFilter = None
) -> Sequence[GUITrace]:
    return _list_traces(session, GUITrace, robot_exception_id)


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
    session: SessionDep, robot_exception_id: RobotExceptionFilter = None
) -> Sequence[ToolTrace]:
    return _list_traces(session, ToolTrace, robot_exception_id)


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
def list_robot_exceptions(session: SessionDep) -> Sequence[RobotException]:
    return session.exec(select(RobotException)).all()


@router.get(
    "/key/{key_id}",
    response_model=Sequence[RobotException],
    summary="List RobotExceptions by RobotKey",
)
def list_robot_exceptions_by_key(
    key_id: UUID,
    session: SessionDep,
) -> Sequence[RobotException]:
    return session.exec(
        select(RobotException).where(RobotException.robot_key_id == key_id)
    ).all()


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
