from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import Engine
from sqlmodel import Session, select

from agent_tools.hooks import AgentLoggingHook
from database.logging.models import AgentTrace, GUITrace, SubAgentTrace
from tests.unit.agent_tools._shared.fakes import (
    FakeBeforeInvocationEvent,
    FakeRegistry,
    is_bound_method_of,
)
from tests.unit.shared.sql_statements import recorded_statements, touching


@pytest.fixture(autouse=True)
def _app_engine(app_engine: Engine) -> Engine:  # pyright: ignore[reportUnusedFunction]
    return app_engine


class MsgEvt:
    """A minimal MessageAddedEvent: the hook only reads its message."""

    def __init__(self, message: dict[str, Any]):
        self.message = message


def _stored_trace(engine: Engine, trace_id: UUID) -> AgentTrace | None:
    with Session(engine) as session:
        return session.get(AgentTrace, trace_id)


def _stored_gui_traces(engine: Engine, agent_trace_id: UUID) -> list[GUITrace]:
    with Session(engine) as session:
        return list(
            session.exec(
                select(GUITrace).where(GUITrace.agent_trace_id == agent_trace_id)
            ).unique()
        )


# ---------------------------------------------------------------------------
# register_hooks
# ---------------------------------------------------------------------------


def test_register_hooks_adds_callbacks():
    registry = FakeRegistry()
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={})

    hook.register_hooks(registry)  # pyright: ignore[reportArgumentType]

    # Should have registered two callbacks
    assert len(registry.callbacks) == 2

    registered_methods = {cb.__name__ for _, cb in registry.callbacks}
    assert "log_start" in registered_methods
    assert "log_message" in registered_methods

    for _, cb in registry.callbacks:
        assert is_bound_method_of(hook, cb)


# ---------------------------------------------------------------------------
# log_start
# ---------------------------------------------------------------------------


def test_log_start_links_the_current_span_to_the_agent_trace():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent_id = uuid4()
    hook = AgentLoggingHook(agent_id=agent_id, invocation_state={})

    with provider.get_tracer("tests").start_as_current_span("invoke_agent"):
        hook.log_start(FakeBeforeInvocationEvent())
    provider.shutdown()

    [span] = exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["agent.id"] == str(agent_id)
    assert span.attributes["agent_trace.id"] == str(hook.agent_trace_id)


def test_log_start_creates_agent_trace(engine: Engine):
    agent_id = uuid4()
    inputs = {"task": "do something"}
    hook = AgentLoggingHook(agent_id=agent_id, invocation_state={"inputs": inputs})

    evt = FakeBeforeInvocationEvent()
    hook.log_start(evt)

    # An AgentTrace must be stored with the generated agent_trace_id
    trace = _stored_trace(engine, hook.agent_trace_id)
    assert isinstance(trace, AgentTrace)
    assert trace.id == hook.agent_trace_id
    assert trace.agent_id == agent_id
    assert trace.inputs == inputs


def test_log_start_creates_subagent_when_parent_id(engine: Engine):
    parent_id = uuid4()
    hook = AgentLoggingHook(
        agent_id=uuid4(), invocation_state={"inputs": {}}, parent_trace_id=parent_id
    )

    hook.log_start(FakeBeforeInvocationEvent())

    assert isinstance(_stored_trace(engine, hook.agent_trace_id), AgentTrace)
    with Session(engine) as session:
        [sub] = session.exec(
            select(SubAgentTrace).where(
                SubAgentTrace.child_trace_id == hook.agent_trace_id
            )
        ).unique()
    assert sub.parent_trace_id == parent_id


# ---------------------------------------------------------------------------
# log_message and update_trace
# ---------------------------------------------------------------------------


def test_log_message_updates_trace_output_and_finished(engine: Engine):
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={"inputs": {}})
    hook.log_start(FakeBeforeInvocationEvent())

    hook.log_message(MsgEvt({"role": "assistant", "content": [{"text": "Hello"}]}))  # pyright: ignore[reportArgumentType]
    hook.log_message(MsgEvt({"role": "user", "content": [{"text": "Hi!"}]}))  # pyright: ignore[reportArgumentType]

    trace = _stored_trace(engine, hook.agent_trace_id)
    assert isinstance(trace, AgentTrace)
    assert trace.messages == hook.messages
    assert [m["content"] for m in trace.messages or []] == [
        [{"text": "Hello"}],
        [{"text": "Hi!"}],
    ]
    assert trace.finished_at is None

    hook.update_trace(finished=True, cost=0.25)

    trace = _stored_trace(engine, hook.agent_trace_id)
    assert isinstance(trace, AgentTrace)
    assert trace.finished_at is not None
    assert trace.cost == 0.25
    assert trace.messages == hook.messages


def test_each_message_is_one_update_and_no_select(engine: Engine):
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={"inputs": {}})
    hook.log_start(FakeBeforeInvocationEvent())

    with recorded_statements(engine) as statements:
        for n in range(5):
            hook.log_message(MsgEvt({"role": "user", "content": [{"text": str(n)}]}))  # pyright: ignore[reportArgumentType]
        hook.update_trace(finished=True)

    assert touching(statements, "SELECT", "agenttrace") == []
    assert len(touching(statements, "UPDATE", "agenttrace")) == 6
    assert len(statements) == 6


def test_update_trace_without_its_trace_raises():
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={})

    with pytest.raises(RuntimeError, match="not found"):
        hook.update_trace()


# ---------------------------------------------------------------------------
# register_gui_trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_gui_trace_saves_a_gui_entry(engine: Engine):
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={})

    await hook.register_gui_trace(
        action_type="click",
        action_content={"x": 10, "y": 20},
        screenshot_bytes=b"\x89PNG",
        success=True,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )

    [gui] = _stored_gui_traces(engine, hook.agent_trace_id)
    assert gui.agent_trace_id == hook.agent_trace_id
    assert gui.action_type == "click"
    assert gui.action_content == {"x": 10, "y": 20}
    assert gui.success is True


@pytest.mark.parametrize(
    "content",
    [
        [{"image": {"format": "jpeg", "source": {"bytes": b"..."}}}],
        [
            {
                "toolResult": {
                    "toolUseId": "t1",
                    "content": [{"image": {"source": {"bytes": b"..."}}}],
                }
            }
        ],
    ],
    ids=["image", "tool_result_image"],
)
@pytest.mark.asyncio
async def test_register_gui_trace_fills_the_pending_image(
    content: list[Any], engine: Engine
):
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={"inputs": {}})
    hook.log_start(FakeBeforeInvocationEvent())
    hook.log_message(MsgEvt({"role": "user", "content": content}))  # pyright: ignore[reportArgumentType]

    def logged_image() -> dict[str, Any]:
        [part] = hook.messages[-1]["content"]
        if "toolResult" in part:
            [part] = part["toolResult"]["content"]
        return part["image"]

    assert logged_image() == {"uuid": "<uuid_pending>"}

    await hook.register_gui_trace(
        action_type="click",
        action_content={},
        screenshot_bytes=b"\x89PNG",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )

    [gui] = _stored_gui_traces(engine, hook.agent_trace_id)
    assert logged_image() == {"uuid": str(gui.screenshot_key)}
    trace = _stored_trace(engine, hook.agent_trace_id)
    assert isinstance(trace, AgentTrace)
    assert trace.messages == hook.messages


@pytest.mark.asyncio
async def test_a_gui_step_is_one_insert_and_one_trace_update(engine: Engine):
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={"inputs": {}})
    hook.log_start(FakeBeforeInvocationEvent())
    image = {"image": {"format": "jpeg", "source": {"bytes": b"..."}}}
    hook.log_message(MsgEvt({"role": "user", "content": [image]}))  # pyright: ignore[reportArgumentType]

    with recorded_statements(engine) as statements:
        await hook.register_gui_trace(
            action_type="click",
            action_content={},
            screenshot_bytes=b"\x89PNG",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )

    assert [s for s in statements if s.startswith("SELECT")] == []
    assert len(touching(statements, "INSERT", "guitrace")) == 1
    assert len(touching(statements, "UPDATE", "agenttrace")) == 1
