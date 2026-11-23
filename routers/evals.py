from __future__ import annotations

import copy
import logging
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from opentelemetry import trace
from opentelemetry.trace import Span
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from sqlmodel import col, select
from strands import Agent as StrandsAgent

import database.general as database
from agent_tools.hooks import RequireToolCalls
from agent_tools.utils import (
    PROCESS_MODEL_STATE_KEY,
    set_continuation,
)
from database.agents.behaviours import GuiAgent, agent_types_of, behaviour_of
from database.agents.models import Agent
from database.keys.models import RobotKey
from database.logging.models import RecoveryContext, RobotException
from observability import attributes as attrs
from observability.correlation import get_request_id
from observability.events import log_event
from observability.exceptions import record_exception
from observability.instrumentation.genai import TracedOpenAIModel
from observability.instrumentation.recovery import record_answer, record_robot_abort
from robot.capabilities import CAPABILITIES_STATE_KEY
from robot.channel import (
    ABORT_CLOSE_CODE,
    CLOSE_SERVER_ERROR,
    AbortedByRobot,
    RobotConnection,
)
from routers.recovery import (  # basedpyright: ignore[reportPrivateUsage]
    CLOSE_INVALID_PAYLOAD,
    INTERNAL_ERROR,
    ROBOT_KEY_HEADER,
    RecoveryWsDomainError,
    _build_answer,
    _check_contract_version_or_raise,
    _decode_payload_or_raise,
    _persist_exception_with_context,
    _record_abort,
    _send_error_and_close,
    _validate_payload_or_raise,
    find_enabled_robot_key,
)
from settings import PROVIDER_API_BASE, PROVIDER_API_KEY, PROVIDER_MODEL
from templates.common import TemplateModel
from templates.recovery_payload import RecoveryPayload

router = APIRouter(prefix="/evals")
tracer = trace.get_tracer(__name__)

logger = logging.getLogger(__name__)

# An invalid payload on this connection; the recovery connection sends
# INVALID_RECOVERY_PAYLOAD. See TECH_DEBT.md, entry 9.
EVAL_VALIDATION_ERROR = "VALIDATION_ERROR"


class ContinuationOutput(TemplateModel):
    """The eval's answer, shaped like ``done.content``; only `success` is reported by the LLM.

    The other fields come from the continuation `set_continuation` stored, as in
    production (`routers.recovery._build_answer`).
    """

    success: bool = Field(
        ...,
        description="Whether the recovery task was fulfilled, so the robot can continue.",
    )
    process_finished: SkipJsonSchema[bool] = False
    continue_from_model_event_id: SkipJsonSchema[str | None] = None
    loop_iteration: SkipJsonSchema[list[int]] = Field(default_factory=list)


CONTINUATION_INFERRER_PROMPT = """
You are a very knowledgeable BPM expert. You are provided with a process model and a recovery trace from a failed instantiation of the process.

This recovery most likely does not cover the entirety of the process, and has left it with some tasks still pending.

Your task is to analyse the following information, decide whether the recovery task was fulfilled, and state it by calling the `set_continuation` tool: with `recovered` false if it wasn't, or with `recovered` true and where the robot continues if it was.

The recovery will most likely not line up with the process model exactly, as the original process had failed. To determine the next task, you must identify or infer the state of the process the recovery has left it in. For example, after the submission of a specific form or after having reached a certain screen in the software.

You will be provided with:
1. The recovery task that was attempted
2. The trace information from the recovery attempt
3. The error information from the failed activity, with its `model_event_id` (the task of the model it performed) when the robot sent it
4. The process model, whose task ids are model event ids

How to name the continuation with `set_continuation(recovered=true, ...)`:
- `model_event_id` is the id of the task in the process model the robot runs next, exactly as the model writes it (e.g. the `id` attribute of a BPMN task). It is never a UI log `event_id`, which counts occurrences in this run, nor a position in the model.
- Naming the errored task means "retry it"; naming an earlier one means "go back".
- `loop_iteration` has one 0-based pass per loop enclosing that task, outermost first, and is [] when the task is in no loop. Read the current passes off the errored entry: a ` #<n>` suffix on its `event_name` per enclosing loop, outermost first. If the recovery moved on to the next pass of a loop, count it.
- If the recovery completed the rest of the process, so the robot has nothing left to run, call it with `process_finished` true.
- If the tool rejects the continuation, correct it as its error says and call it again until it returns `ok`. Only `set_continuation` decides the continuation; your report can't change it.

For example, in a process that loops over pending requests (loop `each_request`) and clicks Approve (task `click_approve`) for each, a robot that failed clicking Approve on the third request (`event_name` "Click Approve #2"):
- couldn't click Approve, the robot should retry it: `set_continuation(recovered=true, process_finished=false, model_event_id="click_approve", loop_iteration=[2])`;
- clicked Approve, the robot continues after it with `back_to_list`: `set_continuation(recovered=true, process_finished=false, model_event_id="back_to_list", loop_iteration=[2])`;
- approved and went back to the list, the robot opens the next request: `set_continuation(recovered=true, process_finished=false, model_event_id="open_request", loop_iteration=[3])`;
- finished every remaining request itself: `set_continuation(recovered=true, process_finished=true, model_event_id=null, loop_iteration=[])`.

If the recovery task has NOT been fulfilled, call `set_continuation(recovered=false, process_finished=false, model_event_id=null, loop_iteration=[])` and report `success` false.

You can't report `success` true without a continuation accepted by `set_continuation`."""


def _get_valid_agent(
    websocket: WebSocket, session: database.SessionDep
) -> Agent | None:
    try:
        agent_id = uuid.UUID(websocket.headers.get("X-AGENT-ID", ""))
    except ValueError:
        return None

    agent = session.exec(
        select(Agent).where(
            Agent.id == agent_id,
            Agent.enabled,
            col(Agent.type).in_(agent_types_of(GuiAgent)),
        )
    ).first()
    if not agent:
        return None

    return agent


def _validate_eval_payload_or_raise(data: Any) -> tuple[RecoveryPayload, str]:
    """The payload and its recovery task, which an eval requires."""
    payload = _validate_payload_or_raise(data, code=EVAL_VALIDATION_ERROR)
    if payload.recovery_task is None:
        raise RecoveryWsDomainError(
            code=EVAL_VALIDATION_ERROR,
            content="recovery_task must be a non-empty string",
            close_code=CLOSE_INVALID_PAYLOAD,
        )
    return payload, payload.recovery_task


async def _run_standalone_recovery(
    robot: RobotConnection,
    payload: dict[str, Any],
    recovery_task: str,
    agent: Agent,
    robot_exception_id: str,
) -> dict[str, Any]:
    variables = payload.get("variables", {})
    ui_log = payload.get("ui_log", [])
    errored_act = payload.get("errored_act", {})
    model = payload.get("model", "")

    # As in production: variables come from code, and `alter_variable` mutates
    # this copy; only `set_continuation` writes the continuation.
    invocation_state: dict[str, Any] = {
        PROCESS_MODEL_STATE_KEY: model,
        "variables": copy.deepcopy(variables),
    }
    gui_state: dict[str, Any] = {
        "websocket": robot,
        "robot_exception_id": robot_exception_id,
        "variables": invocation_state["variables"],
        # Validated with the payload; absent means vision only.
        CAPABILITIES_STATE_KEY: payload.get("capabilities"),
    }

    gui_agent = behaviour_of(agent)
    if not isinstance(gui_agent, GuiAgent):
        raise TypeError(f"The eval agent '{agent.name}' is not a GUI agent.")
    run = await gui_agent.drive(
        gui_state,
        {
            "task": recovery_task,
            "variables": variables,
            "ui_log": ui_log,
            "errored_act": errored_act,
            "model": model,
        },
        contain_errors=True,
    )
    trace_text = run.error if run.error is not None else str(run.transcript())

    agent_model = TracedOpenAIModel(
        client_args={"api_key": PROVIDER_API_KEY, "base_url": PROVIDER_API_BASE},
        model_id=PROVIDER_MODEL,
    )

    # The same rule as the production error agent: `set_continuation` is a
    # required tool, so a run that skips it is sent back to call it.
    continuation_inferrer = StrandsAgent(
        model=agent_model,
        system_prompt=CONTINUATION_INFERRER_PROMPT,
        tools=[set_continuation],
        hooks=[RequireToolCalls([set_continuation.tool_name])],
        callback_handler=None,
    )

    inferrer_prompt = f"""Recovery Task: {recovery_task}

Trace Information:
{trace_text}

Error Information:
{errored_act}

Process Model:
{model}"""

    _ = await continuation_inferrer.invoke_async(
        inferrer_prompt, invocation_state=invocation_state
    )
    inference = await continuation_inferrer.invoke_async(
        "Given our conversation so far, report whether the recovery task was "
        "fulfilled using structured output with the given tool.",
        structured_output_model=ContinuationOutput,
        invocation_state=invocation_state,
    )
    report = inference.structured_output
    if not isinstance(report, ContinuationOutput):
        raise TypeError("The continuation inferrer did not return a structured report.")

    # Scored against the same answer production sends in `done.content`.
    answer, failure_reason = _build_answer(report, invocation_state)
    span = trace.get_current_span()
    span.set_attribute(attrs.RECOVERY_SUCCESS, answer.success)
    record_answer(span, answer, failure_reason)
    return answer.model_dump()


@router.websocket("/recovery/ws")
async def handle_recovery_eval(
    websocket: WebSocket, session: database.SessionDep
) -> None:
    robot_key = find_enabled_robot_key(session, websocket.headers.get(ROBOT_KEY_HEADER))
    agent = _get_valid_agent(websocket, session)
    if robot_key and agent:
        await websocket.accept()
    else:
        raise WebSocketDisconnect(code=1008, reason="Invalid robot key or agent id")

    with tracer.start_as_current_span(
        attrs.SPAN_RECOVERY_EVAL,
        attributes={
            attrs.ROBOT_KEY_ID: str(robot_key.id),
            attrs.AGENT_ID: str(agent.id),
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        async with RobotConnection(websocket) as robot:
            await _run_eval_turn(robot, session, robot_key, agent, span)


async def _run_eval_turn(
    robot: RobotConnection,
    session: database.SessionDep,
    robot_key: RobotKey,
    agent: Agent,
    span: Span,
) -> None:
    """One eval: payload, recovery task, record, GUI agent run, result.

    As in production, an abort (before or during the run) cancels it, fails the
    recovery with its reason and sends no result.
    """
    exception: RobotException | None = None
    try:
        data = _decode_payload_or_raise(await robot.receive_text())
        _check_contract_version_or_raise(data)
        payload, recovery_task = _validate_eval_payload_or_raise(data)

        recovery_context = RecoveryContext.from_payload(payload=payload)
        exception = _persist_exception_with_context(
            session, data, robot_key, recovery_context
        )

        result = await robot.run(
            _run_standalone_recovery(
                robot, data, recovery_task, agent, str(exception.id)
            )
        )

        await robot.send_json(
            {
                "type": "eval_result",
                "content": result,
                "exception_id": str(exception.id),
                "request_id": get_request_id(),
            }
        )
        await robot.close()

        exception.infered_success = result["success"]
        session.add(exception)
        session.commit()
    except AbortedByRobot as exc:
        if exception is not None:
            _record_abort(session, exception, exc.abort)
        record_robot_abort(span, exc.abort)
        await robot.close(ABORT_CLOSE_CODE)
    except WebSocketDisconnect:
        pass
    except RecoveryWsDomainError as exc:
        # An unusable eval request: a warning, not an error.
        log_event(
            logger,
            attrs.EVENT_RECOVERY_EVAL_INVALID,
            {attrs.ERROR_TYPE: exc.code},
            logging.WARNING,
        )
        await _send_error_and_close(robot, exc.code, exc.content, exc.close_code)
    except Exception as exc:
        logger.exception("Error during eval")
        record_exception(span, exc, escaped=False)
        await _send_error_and_close(
            robot, INTERNAL_ERROR, "Internal server error", CLOSE_SERVER_ERROR
        )
