from dataclasses import dataclass

from pydantic import ValidationError
from pydantic_core import InitErrorDetails
from strands import ToolContext, tool

from agent_tools.process_model import (
    Loop,
    ModelEventError,
    check_model_event,
    describe_model_event,
    enclosing_loops,
)

CONTINUATION_STATE_KEY = "continuation"


@dataclass(frozen=True)
class Continuation:
    """Where the robot continues after the recovery, as validated by `set_continuation`.

    Attributes:
        process_finished: The recovery completed the rest of the process.
        model_event_id: The task of the model the robot runs next; `None`
            when the process is finished.
        loop_iteration: 0-based passes of the loops enclosing that task,
            outermost first.
        loop_depth_checked: Whether the length of `loop_iteration` was checked
            against the model. It isn't when the model is not BPMN or its loops
            can't be read.
    """

    process_finished: bool
    model_event_id: str | None
    loop_iteration: tuple[int, ...]
    loop_depth_checked: bool


def _invalid(field: str, value: object, message: str) -> ValidationError:
    """Build the ValidationError the agent sees; `LimitToolCounts` refunds the call."""
    details: InitErrorDetails = {
        "type": "value_error",
        "loc": (field,),
        "input": value,
        "ctx": {"error": ValueError(message)},
    }
    return ValidationError.from_exception_data("set_continuation", [details])


def _describe_loops(loops: list[Loop]) -> str:
    return ", ".join(
        f"{loop.id} ({loop.name})" if loop.name else loop.id for loop in loops
    )


@tool(
    context=True,
    description=(
        "Set where the RPA robot continues after the recovery. This is the only way to "
        "name the continuation, and it must succeed before you finish.\n"
        "- process_finished: true if the recovery completed the rest of the process, so "
        "the robot has nothing left to run; model_event_id and loop_iteration are then "
        "ignored.\n"
        "- model_event_id: the id of the task in `model` the robot runs next (a model "
        "event id, never a UI log event_id). Naming the errored task means 'retry it'; "
        "naming an earlier one means 'go back'.\n"
        "- loop_iteration: one 0-based pass per loop enclosing that task, outermost "
        "first; [] when the task is in no loop.\n"
        "If the continuation is invalid, the error says what is wrong and what is "
        "expected: correct it and call again."
    ),
)
def set_continuation(
    process_finished: bool,
    model_event_id: str | None,
    loop_iteration: list[int],
    tool_context: ToolContext,
) -> str:
    state = tool_context.invocation_state

    if process_finished:
        state[CONTINUATION_STATE_KEY] = Continuation(
            process_finished=True,
            model_event_id=None,
            loop_iteration=(),
            loop_depth_checked=True,
        )
        return "ok: the process is finished; the robot has nothing left to run."

    model = state.get("model")
    if not isinstance(model, str):
        raise RuntimeError(
            "The process model must be provided in the invocation state."
        )

    if model_event_id is None or not model_event_id.strip():
        raise _invalid(
            "model_event_id",
            model_event_id,
            "model_event_id is required unless process_finished is true.",
        )
    try:
        check_model_event(model, model_event_id)
    except ModelEventError as exc:
        raise _invalid("model_event_id", model_event_id, str(exc)) from exc

    if any(p < 0 for p in loop_iteration):
        raise _invalid(
            "loop_iteration",
            loop_iteration,
            f"loop passes are 0-based and can't be negative, got {loop_iteration}.",
        )

    loops = enclosing_loops(model, model_event_id)
    if loops is not None and len(loops) != len(loop_iteration):
        if loops:
            expected = (
                f"{model_event_id} is inside {len(loops)} "
                f"loop{'s' if len(loops) > 1 else ''} ({_describe_loops(loops)}, "
                f"outermost first), so loop_iteration needs {len(loops)} "
                f"pass{'es' if len(loops) > 1 else ''}"
            )
        else:
            expected = f"{model_event_id} is in no loop, so loop_iteration must be []"
        raise _invalid(
            "loop_iteration", loop_iteration, f"{expected}, got {loop_iteration}."
        )

    state[CONTINUATION_STATE_KEY] = Continuation(
        process_finished=False,
        model_event_id=model_event_id,
        loop_iteration=tuple(loop_iteration),
        loop_depth_checked=loops is not None,
    )

    event = describe_model_event(model, model_event_id)
    target = (
        f"{model_event_id} ({event.name})" if event and event.name else model_event_id
    )
    if loops is None:
        where = (
            f"with loop_iteration {loop_iteration} as given (the loop depth can't be "
            "read from this model)"
        )
    elif loops:
        passes = ", ".join(
            f"{loop.id} pass {p}" for loop, p in zip(loops, loop_iteration, strict=True)
        )
        where = f"inside {_describe_loops(loops)}: {passes}"
    else:
        where = "in no loop"
    return f"ok: the robot continues from {target}, {where}."
