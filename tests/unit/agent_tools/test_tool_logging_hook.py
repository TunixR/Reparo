from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine
from sqlmodel import Session
from strands.types.tools import ToolResult

from agent_tools.hooks import ToolLoggingHook
from database.logging.models import ToolTrace
from tests.unit.agent_tools._shared.fakes import (
    FakeAfterToolCallEvent,
    FakeBeforeToolCallEvent,
)
from tests.unit.shared.sql_statements import recorded_statements, touching


@pytest.fixture(autouse=True)
def _app_engine(app_engine: Engine) -> Engine:  # pyright: ignore[reportUnusedFunction]
    return app_engine


def _stored(engine: Engine, trace_id: UUID) -> ToolTrace | None:
    with Session(engine) as session:
        return session.get(ToolTrace, trace_id)


def test_log_tool_call_inserts_trace_for_tracked_tool(engine: Engine):
    agent_trace_id = uuid4()
    tool_id = uuid4()
    tools = {"my_tool": tool_id}
    hook = ToolLoggingHook(agent_trace_id=agent_trace_id, tools=tools)

    evt = FakeBeforeToolCallEvent("my_tool", input_value={"key": "val"})
    hook.log_tool_call(evt)

    assert isinstance(hook.trace_id, UUID)
    trace_obj = _stored(engine, hook.trace_id)
    assert isinstance(trace_obj, ToolTrace)
    assert trace_obj.agent_trace_id == agent_trace_id
    assert trace_obj.tool_id == tool_id
    assert trace_obj.input == {"key": "val"}
    assert trace_obj.finished_at is None


def test_log_tool_call_ignored_for_untracked_tool(engine: Engine):
    agent_trace_id = uuid4()
    tools = {"tracked_tool": uuid4()}
    hook = ToolLoggingHook(agent_trace_id=agent_trace_id, tools=tools)

    with recorded_statements(engine) as statements:
        hook.log_tool_call(FakeBeforeToolCallEvent("other_tool", input_value="x"))

    # No trace should be created and no trace_id set
    assert not hasattr(hook, "trace_id")
    assert statements == []


def test_log_tool_response_success_updates_trace(engine: Engine):
    agent_trace_id = uuid4()
    tool_id = uuid4()
    hook = ToolLoggingHook(agent_trace_id=agent_trace_id, tools={"tool_x": tool_id})

    call_evt = FakeBeforeToolCallEvent("tool_x", input_value="input-str")
    hook.log_tool_call(call_evt)

    result: ToolResult = {
        "status": "success",
        "content": [{"text": "result text"}],
        "toolUseId": "",
    }
    resp_evt = FakeAfterToolCallEvent("tool_x", exception=None, result=result)
    hook.log_tool_response(resp_evt)

    trace_obj = _stored(engine, hook.trace_id)
    assert isinstance(trace_obj, ToolTrace)
    assert trace_obj.success is True
    assert trace_obj.output == "result text"
    assert trace_obj.finished_at is not None


def test_log_tool_response_failure_updates_trace_with_exception(engine: Engine):
    agent_trace_id = uuid4()
    tool_id = uuid4()
    hook = ToolLoggingHook(agent_trace_id=agent_trace_id, tools={"tool_y": tool_id})

    call_evt = FakeBeforeToolCallEvent("tool_y", input_value={"param": 1})
    hook.log_tool_call(call_evt)

    resp_evt = FakeAfterToolCallEvent(
        "tool_y", exception=RuntimeError("boom"), result=None
    )
    hook.log_tool_response(resp_evt)

    trace_obj = _stored(engine, hook.trace_id)
    assert isinstance(trace_obj, ToolTrace)
    assert trace_obj.success is False
    assert "boom" in (trace_obj.output or "")
    assert trace_obj.finished_at is not None


def test_a_tool_call_is_one_insert_and_one_update(engine: Engine):
    hook = ToolLoggingHook(agent_trace_id=uuid4(), tools={"tool_x": uuid4()})

    with recorded_statements(engine) as statements:
        hook.log_tool_call(FakeBeforeToolCallEvent("tool_x", input_value={}))
        hook.log_tool_response(FakeAfterToolCallEvent("tool_x", exception=None))

    assert touching(statements, "SELECT", "tooltrace") == []
    assert len(touching(statements, "INSERT", "tooltrace")) == 1
    assert len(touching(statements, "UPDATE", "tooltrace")) == 1


def test_log_tool_response_without_its_trace_raises():
    hook = ToolLoggingHook(agent_trace_id=uuid4(), tools={"tool_x": uuid4()})
    hook.trace_id = uuid4()

    with pytest.raises(RuntimeError, match="not found"):
        hook.log_tool_response(FakeAfterToolCallEvent("tool_x", exception=None))
