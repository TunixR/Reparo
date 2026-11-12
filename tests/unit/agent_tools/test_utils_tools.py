"""Tests for the `set_continuation` agent tool.

The tool runs through Strands' own call path (`DecoratedFunctionTool.stream`),
so the error the agent sees and the exception `LimitToolCounts` receives are
the ones a real agent loop produces.
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from pydantic import ValidationError
from strands.types.events import ToolResultEvent

from agent_tools.hooks import LimitToolCounts
from agent_tools.utils import (
    CONTINUATION_STATE_KEY,
    PROCESS_MODEL_STATE_KEY,
    Continuation,
    set_continuation,
)
from tests.unit.agent_tools._shared.fakes import (
    FakeAfterToolCallEvent,
    FakeBeforeToolCallEvent,
)
from tests.unit.agent_tools._shared.process_models import (
    APPROVAL_XOR_MODEL,
    COMPACT_MODEL,
    NESTED_LOOPS_MODEL,
    SPENDING_APPROVAL_MODEL,
)

TOOL_NAME = "set_continuation"


def _call(state: dict[str, Any], **tool_input: Any) -> ToolResultEvent:
    """Call the tool as an agent would and return its result event."""

    async def run() -> ToolResultEvent:
        last: Any = None
        tool_use = {"toolUseId": "t1", "name": TOOL_NAME, "input": tool_input}
        async for event in set_continuation.stream(tool_use, state):  # pyright: ignore[reportArgumentType]
            last = event
        assert isinstance(last, ToolResultEvent)
        return last

    return asyncio.run(run())


def _state(model: str) -> dict[str, Any]:
    return {"agent": MagicMock(), PROCESS_MODEL_STATE_KEY: model}


def _text(result: ToolResultEvent) -> str:
    return result.tool_result["content"][0].get("text", "")


def _assert_rejected_and_refunded(
    state: dict[str, Any], result: ToolResultEvent, *expected: str
) -> None:
    """The call failed with a ValidationError, stored nothing and was refunded."""
    assert result.tool_result["status"] == "error"
    assert isinstance(result.exception, ValidationError)
    for text in expected:
        assert text in _text(result)
    assert CONTINUATION_STATE_KEY not in state

    limiter = LimitToolCounts(max_tool_counts={TOOL_NAME: 1})
    limiter.intercept_tool(FakeBeforeToolCallEvent(TOOL_NAME))
    limiter.intercept_response(
        FakeAfterToolCallEvent(TOOL_NAME, result.exception, result.tool_result)
    )
    assert limiter.tool_counts[TOOL_NAME] == 0


# ---------------------------------------------------------------------------
# Rejected continuations: the agent is told why and retries
# ---------------------------------------------------------------------------


def test_rejects_an_unknown_model_event_id():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state, process_finished=False, model_event_id="click_aprove", loop_iteration=[2]
    )

    _assert_rejected_and_refunded(
        state,
        result,
        "'click_aprove' is not a task of the model",
        "click_approve (Click Approve)",
    )


def test_rejects_a_ui_log_event_id_that_is_not_a_task_id():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state, process_finished=False, model_event_id="8", loop_iteration=[2]
    )

    _assert_rejected_and_refunded(state, result, "'8' is not a task of the model")


def test_rejects_a_loop_iteration_of_the_wrong_length():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state,
        process_finished=False,
        model_event_id="click_approve",
        loop_iteration=[1, 2],
    )

    _assert_rejected_and_refunded(
        state,
        result,
        "click_approve is inside 1 loop (each_request (For each request_id in pending), "
        "outermost first), so loop_iteration needs 1 pass, got [1, 2]",
    )


def test_rejects_passes_for_an_event_in_no_loop():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state,
        process_finished=False,
        model_event_id="open_requests",
        loop_iteration=[0],
    )

    _assert_rejected_and_refunded(
        state, result, "open_requests is in no loop, so loop_iteration must be []"
    )


def test_rejects_a_negative_pass():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state,
        process_finished=False,
        model_event_id="click_approve",
        loop_iteration=[-1],
    )

    _assert_rejected_and_refunded(state, result, "can't be negative")


def test_rejects_a_missing_model_event_id_when_the_process_is_not_finished():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state, process_finished=False, model_event_id=None, loop_iteration=[]
    )

    _assert_rejected_and_refunded(state, result, "model_event_id is required")


# ---------------------------------------------------------------------------
# Stored continuations
# ---------------------------------------------------------------------------


def test_stores_a_task_inside_an_xor_branch_whose_ui_log_ids_differ():
    # The run logged UI log events 1 and 2 and failed on 3 (click_approve);
    # the recovery decided the request must be rejected.
    state = _state(APPROVAL_XOR_MODEL)

    result = _call(
        state, process_finished=False, model_event_id="click_reject", loop_iteration=[]
    )

    assert result.tool_result["status"] == "success"
    assert result.exception is None
    assert _text(result) == (
        "ok: the robot continues from click_reject (Click Reject), in no loop."
    )
    assert state[CONTINUATION_STATE_KEY] == Continuation(
        process_finished=False,
        model_event_id="click_reject",
        loop_iteration=(),
    )


def test_stores_both_passes_of_nested_loops():
    state = _state(NESTED_LOOPS_MODEL)

    result = _call(
        state,
        process_finished=False,
        model_event_id="click_approve",
        loop_iteration=[1, 2],
    )

    assert _text(result) == (
        "ok: the robot continues from click_approve (Click Approve), inside "
        "each_approver (For each approver), each_request (For each request of the "
        "approver): each_approver pass 1, each_request pass 2."
    )
    assert state[CONTINUATION_STATE_KEY].loop_iteration == (1, 2)


def test_stores_the_passes_as_given_for_a_compact_model_of_unknown_depth():
    state = _state(COMPACT_MODEL)

    result = _call(
        state,
        process_finished=False,
        model_event_id="click_approve",
        loop_iteration=[2],
    )

    assert result.tool_result["status"] == "success"
    assert "can't be read from this model" in _text(result)
    assert state[CONTINUATION_STATE_KEY] == Continuation(
        process_finished=False,
        model_event_id="click_approve",
        loop_iteration=(2,),
    )


@pytest.mark.parametrize(
    ("model", "loop_iteration", "unchecked"),
    [(COMPACT_MODEL, [2], True), (SPENDING_APPROVAL_MODEL, [2], False)],
)
def test_traces_an_unchecked_loop_depth_on_the_tool_span(
    model: str, loop_iteration: list[int], unchecked: bool
):
    tracer = TracerProvider().get_tracer("tests.set_continuation")

    with tracer.start_as_current_span("execute_tool set_continuation") as span:
        _ = _call(
            _state(model),
            process_finished=False,
            model_event_id="click_approve",
            loop_iteration=loop_iteration,
        )

    assert isinstance(span, ReadableSpan)
    events = [event.name for event in span.events]
    assert ("loop_depth_unchecked" in events) is unchecked


def test_stores_a_finished_process():
    state = _state(SPENDING_APPROVAL_MODEL)

    result = _call(
        state, process_finished=True, model_event_id="ignored", loop_iteration=[9]
    )

    assert result.tool_result["status"] == "success"
    assert state[CONTINUATION_STATE_KEY] == Continuation(
        process_finished=True,
        model_event_id=None,
        loop_iteration=(),
    )


def test_a_later_valid_call_replaces_a_rejected_one():
    state = _state(SPENDING_APPROVAL_MODEL)

    rejected = _call(
        state, process_finished=False, model_event_id="8", loop_iteration=[2]
    )
    accepted = _call(
        state,
        process_finished=False,
        model_event_id="click_approve",
        loop_iteration=[2],
    )

    assert rejected.tool_result["status"] == "error"
    assert accepted.tool_result["status"] == "success"
    assert state[CONTINUATION_STATE_KEY].model_event_id == "click_approve"


def test_fails_without_a_model_in_the_invocation_state():
    state: dict[str, Any] = {"agent": MagicMock()}

    result = _call(
        state, process_finished=False, model_event_id="click_approve", loop_iteration=[]
    )

    assert result.tool_result["status"] == "error"
    assert isinstance(result.exception, RuntimeError)
    assert CONTINUATION_STATE_KEY not in state


@pytest.mark.parametrize(
    "field", ["process_finished", "model_event_id", "loop_iteration"]
)
def test_every_argument_is_required(field: str):
    schema = set_continuation.tool_spec["inputSchema"]["json"]

    assert field in schema["required"]
