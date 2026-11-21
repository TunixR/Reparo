import copy
import json
from contextlib import suppress
from typing import Any
from uuid import UUID

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from opentelemetry import trace
from pydantic import BaseModel, ValidationError
from sqlmodel import select

import database.general as database
from agent_tools.utils import (
    PROCESS_MODEL_STATE_KEY,
    declared_not_recovered,
    stored_continuation,
)
from database.agent_graphs.models import AgentGraph
from database.agent_graphs.runner import (
    AGENT_GRAPH_CONTEXT_KEY,
    AgentGraphContext,
    GraphOutcome,
)
from database.keys.models import RobotKey
from database.logging.agent_graph_tracing import AgentGraphRunTrace
from database.logging.models import RecoveryContext, RobotException
from observability import attributes as attrs
from observability.instrumentation import recovery as telemetry
from robot.capabilities import CAPABILITIES_STATE_KEY
from robot.channel import (
    ABORT_CLOSE_CODE,
    CLOSE_SERVER_ERROR,
    AbortedByRobot,
    KeepAliveFailed,
    RobotAbort,
    RobotConnection,
)
from security.utils import robot_key_hash
from templates.common import TemplateModel
from templates.gateway import ResponseToRPA
from templates.recovery_payload import CONTRACT_VERSION, RecoveryPayload

router = APIRouter(prefix="/recovery")
tracer = trace.get_tracer(__name__)


ROBOT_KEY_HEADER = "X-ROBOT-KEY"


INVALID_RECOVERY_PAYLOAD = "INVALID_RECOVERY_PAYLOAD"
UNSUPPORTED_CONTRACT_VERSION = "UNSUPPORTED_CONTRACT_VERSION"
NO_AGENT_GRAPH = "NO_AGENT_GRAPH"
INTERNAL_ERROR = "INTERNAL_ERROR"

# Websocket close code (RFC 6455): unsupported data.
CLOSE_INVALID_PAYLOAD = 1003


class RecoveryWsDomainError(Exception):
    """A turn ending that is reported to the robot as an ``error`` message."""

    def __init__(self, code: str, content: str, close_code: int):
        super().__init__(content)
        self.code = code
        self.content = content
        self.close_code = close_code


class RecoveryWsErrorResponse(TemplateModel):
    type: str = "error"
    code: str
    content: str


def find_enabled_robot_key(
    session: database.SessionDep, key_raw: str | None
) -> RobotKey | None:
    """Return the enabled robot key matching a raw key, if any."""
    if not key_raw:
        return None

    robot_key = session.exec(
        select(RobotKey).where(RobotKey.key_hash == robot_key_hash(key_raw))
    ).first()
    if not robot_key or not robot_key.enabled:
        return None

    return robot_key


def _decode_payload_or_raise(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecoveryWsDomainError(
            code=INVALID_RECOVERY_PAYLOAD,
            content=f"Recovery payload is not valid JSON: {exc.msg}",
            close_code=CLOSE_INVALID_PAYLOAD,
        ) from exc


def _check_contract_version_or_raise(data: Any) -> None:
    """Refuse a decoded payload that doesn't speak Reparo's Contract version.

    Runs before anything else reads the payload. Only the JSON integer equal to
    `CONTRACT_VERSION` passes: a bool, a float or a string is not a version.
    """
    sent = data.get("contract_version") if isinstance(data, dict) else None
    if type(sent) is int and sent == CONTRACT_VERSION:
        return
    received = "none" if sent is None else json.dumps(sent)
    raise RecoveryWsDomainError(
        code=UNSUPPORTED_CONTRACT_VERSION,
        content=(
            f"Reparo speaks contract version {CONTRACT_VERSION}; "
            f"the payload sent {received}."
        ),
        close_code=CLOSE_INVALID_PAYLOAD,
    )


def _build_recovery_context_or_raise(data: Any):
    try:
        payload = RecoveryPayload.model_validate(data)
        return RecoveryContext.from_payload(payload=payload)
    except ValidationError as exc:
        raise RecoveryWsDomainError(
            code=INVALID_RECOVERY_PAYLOAD,
            content=str(exc),
            close_code=CLOSE_INVALID_PAYLOAD,
        ) from exc


async def _send_error_and_close(
    robot: RobotConnection, code: str, content: str, close_code: int
) -> None:
    # Best effort: the robot may be why the turn is ending.
    with suppress(WebSocketDisconnect):
        await robot.send_json(
            RecoveryWsErrorResponse(code=code, content=content).model_dump()
        )
    # The reason is capped at 123 bytes by the protocol; the code always fits.
    await robot.close(code=close_code, reason=code)


def _persist_exception_with_context(
    session: database.SessionDep,
    data: dict[str, Any],
    robot_key: RobotKey,
    recovery_context: RecoveryContext,
):
    exception = RobotException(
        exception_details=data,
        robot_key_id=robot_key.id,
    )
    exception.recovery_context = recovery_context
    session.add(exception)
    session.commit()
    session.refresh(exception)
    return exception


def _record_abort(
    session: database.SessionDep, exception: RobotException, abort: RobotAbort
) -> None:
    """Store a robot abort on its recovery, which ends as failed."""
    exception.abort_code = abort.error_code
    exception.abort_description = abort.error_description
    exception.infered_success = False
    session.add(exception)
    session.commit()


def _build_answer(
    last_output: BaseModel | None,
    invocation_state: dict[str, Any],
    *,
    graph_failed: bool = False,
) -> tuple[ResponseToRPA, str | None]:
    """Build ``done.content``: ``success`` from the last output, the rest from the state.

    ``success`` is the last output's ``success`` field, false when it is missing. A
    success also needs a continuation from `set_continuation`; a failure carries no
    continuation.

    Returns:
        The answer, and why it had to be sent as a failure: ``agent_graph_failed``
        if the graph hit a limit, ``missing_continuation`` if a reported success
        came without `set_continuation`, or ``not_recovered`` if it was called with
        ``recovered`` false.
    """
    continuation = stored_continuation(invocation_state)
    failure_reason = None
    success = getattr(last_output, "success", False) is True
    if graph_failed:
        success = False
        failure_reason = attrs.EVENT_RECOVERY_AGENT_GRAPH_FAILED
    elif success and continuation is None:
        success = False
        failure_reason = (
            attrs.EVENT_RECOVERY_NOT_RECOVERED
            if declared_not_recovered(invocation_state)
            else attrs.EVENT_RECOVERY_MISSING_CONTINUATION
        )
    answer = ResponseToRPA(
        success=success,
        variables=invocation_state["variables"],
        **(continuation.model_dump() if success and continuation else {}),
    )
    return answer, failure_reason


def _recovery_inputs(recovery_context: RecoveryContext) -> dict[str, Any]:
    """The recovery inputs, from the recovery payload."""
    ui_log = [
        {
            "case_id": entry.case_id,
            "activity_id": entry.activity_id,
            "activity_name": entry.activity_name,
            "event_id": entry.event_id,
            "event_name": entry.event_name,
            "model_event_id": entry.model_event_id,
            "action_type": entry.action_type,
            "application": entry.application,
            "input": entry.input,
            "ui_element_target": entry.ui_element_target,
            "ui_group": entry.ui_group,
            "timestamp": entry.timestamp,
            "previous_state": entry.previous_state,
            "current_state": entry.current_state,
        }
        for entry in recovery_context.ui_log_entries
    ]

    errored_act = None
    if recovery_context.errored_activity:
        act = recovery_context.errored_activity
        errored_act = {
            "case_id": act.case_id,
            "activity_id": act.activity_id,
            "activity_name": act.activity_name,
            "event_id": act.event_id,
            "event_name": act.event_name,
            "model_event_id": act.model_event_id,
            "action_type": act.action_type,
            "application": act.application,
            "input": act.input,
            "error_code": act.error_code,
            "error_description": act.error_description,
        }

    return {
        "task_name": recovery_context.task_name,
        "platform": recovery_context.platform,
        "os": recovery_context.os,
        "variables": recovery_context.variables,
        "ui_log": ui_log,
        "errored_act": errored_act,
        "model": recovery_context.model,
    }


async def _run_agent_graph(
    session: database.SessionDep,
    graph_id: UUID,
    robot: RobotConnection,
    exception: RobotException,
    recovery_context: RecoveryContext,
    run_trace: AgentGraphRunTrace,
) -> tuple[GraphOutcome, dict[str, Any]]:
    """Run the agent graph; returns its outcome and the final invocation state."""
    invocation_state: dict[str, Any] = {
        "websocket": robot,
        "robot_exception_id": exception.id,
        PROCESS_MODEL_STATE_KEY: recovery_context.model,
        "variables": copy.deepcopy(recovery_context.variables),
        CAPABILITIES_STATE_KEY: recovery_context.capabilities,
        AGENT_GRAPH_CONTEXT_KEY: AgentGraphContext(
            recovery_inputs=_recovery_inputs(recovery_context), trace=run_trace
        ),
    }
    graph = session.get(AgentGraph, graph_id)
    if graph is None:
        raise LookupError(f"The agent graph {graph_id} is gone.")
    outcome = await graph.run(invocation_state)
    return outcome, invocation_state


@router.websocket("/robot_exception/ws")
async def handle_robot_exception(websocket: WebSocket, session: database.SessionDep):
    """
    Passes the exception to the robot exception handler for processing.
    """
    robot_key = find_enabled_robot_key(session, websocket.headers.get(ROBOT_KEY_HEADER))
    if not robot_key:
        telemetry.auth_failed(websocket.client)
        raise WebSocketDisconnect(code=1008, reason="Invalid robot key")

    await websocket.accept()

    with telemetry.recovery_turn(tracer, robot_key.id) as turn:
        async with RobotConnection(websocket) as robot:
            await _run_recovery_turn(robot, session, robot_key, turn)


async def _run_recovery_turn(
    robot: RobotConnection,
    session: database.SessionDep,
    robot_key: RobotKey,
    turn: telemetry.RecoveryTurnTelemetry,
) -> None:
    """One recovery: payload, record, agent graph run, answer.

    ``done`` means a recovery was instantiated and answered; every other ending
    is an ``error`` with a code, except a robot that aborted or went away.
    """
    exception: RobotException | None = None
    try:
        # One payload per connection. Read the raw text so its size is known
        # without re-serializing it. An abort instead of a payload just closes.
        raw = await robot.receive_text()
        data = _decode_payload_or_raise(raw)
        turn.payload_received(raw, data)
        _check_contract_version_or_raise(data)

        # Validate before looking up the agent graph, so a malformed payload is
        # reported as such even on a misconfigured server.
        recovery_context = _build_recovery_context_or_raise(data)
        turn.context_built(recovery_context)

        graph_id = AgentGraph.default_recovery_id(session)
        if graph_id is None:
            raise RecoveryWsDomainError(
                code=NO_AGENT_GRAPH,
                content="No enabled default recovery agent graph in the database.",
                close_code=CLOSE_SERVER_ERROR,
            )
        turn.agent_graph_selected(graph_id)

        with turn.persisting():
            exception = _persist_exception_with_context(
                session, data, robot_key, recovery_context
            )
        turn.persisted(exception.id)

        # The agent graph run is a child span of this turn. It is cancelled as
        # soon as the robot aborts or goes away.
        run_trace = AgentGraphRunTrace.open(graph_id, exception.id)
        try:
            outcome, invocation_state = await robot.run(
                _run_agent_graph(
                    session,
                    graph_id,
                    robot,
                    exception,
                    recovery_context,
                    run_trace,
                )
            )
            answer, failure_reason = _build_answer(
                outcome.output,
                invocation_state,
                graph_failed=outcome.failure_reason is not None,
            )
            run_trace.close(outcome, answer)
        except BaseException as exc:
            run_trace.fail(exc)
            raise
        turn.answered(answer, failure_reason)
        success = answer.success

        await robot.send_json(
            {"type": "done", "content": answer.model_dump(), "id": str(exception.id)}
        )
        await robot.close()

        exception.infered_success = success
        session.add(exception)
        session.commit()
        turn.completed(success)
    except AbortedByRobot as exc:
        # No ``done``: the recovery, if there is one, fails with the robot's reason.
        if exception is not None:
            _record_abort(session, exception, exc.abort)
        turn.robot_aborted(exc.abort)
        await robot.close(ABORT_CLOSE_CODE)
    except WebSocketDisconnect:
        turn.client_disconnected()
    except RecoveryWsDomainError as exc:
        if exc.code == NO_AGENT_GRAPH:
            turn.no_agent_graph()
        else:
            turn.invalid_payload(exc)
        await _send_error_and_close(robot, exc.code, exc.content, exc.close_code)
    except Exception as exc:  # noqa: BLE001 - reported to the client and the span
        if isinstance(exc, KeepAliveFailed):
            turn.keep_alive_failed(exc.__cause__ or exc)
        turn.failed(exc)
        await _send_error_and_close(
            robot, INTERNAL_ERROR, "Internal server error", CLOSE_SERVER_ERROR
        )


@router.post("/report_result/{recovery_id}")
async def report_recovery_result(
    request: Request, recovery_id: str, session: database.SessionDep
):
    # Rejections below are logged by the global HTTP exception handler.
    try:
        recovery_uuid = UUID(recovery_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid recovery ID format. Must be a UUID.",
        ) from exc

    key_raw = request.headers.get(ROBOT_KEY_HEADER)
    if not key_raw:
        raise HTTPException(status_code=401, detail="No robot key provided")

    robot_key = find_enabled_robot_key(session, key_raw)
    if not robot_key:
        raise HTTPException(status_code=403, detail="Invalid robot key")

    body = await request.json()
    success = body.get("success", False)

    exception = session.get(RobotException, recovery_uuid)
    if not exception:
        raise HTTPException(status_code=404, detail="Recovery ID not found")

    exception.reported_success = success
    session.add(exception)
    session.commit()
    trace.get_current_span().set_attributes(
        {
            attrs.RECOVERY_ID: str(recovery_uuid),
            attrs.ROBOT_KEY_ID: str(robot_key.id),
            attrs.RECOVERY_SUCCESS: bool(success),
        }
    )
    return Response(status_code=204)
