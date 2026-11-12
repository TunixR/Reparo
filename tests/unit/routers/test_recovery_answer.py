"""The answer sent in ``done``: a validated continuation, and variables from code.

The gateway and error agents are real ``Agent`` rows run by real Strands agents
over scripted models, so ``set_continuation``, its required-tool reminders and the
invocation state run as they do in production.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlmodel import Session, select

import routers.recovery as recovery_module
from agent_tools.hooks import MAX_REQUIRED_TOOL_REMINDERS
from database.agents.models import (
    Agent,
    AgentTool,
    AgentType,
    Argument,
    SubAgent,
)
from database.logging.models import RobotException
from database.provider.models import Router
from database.tools.models import Tool
from tests.unit.agent_tools._shared.process_models import (
    COMPACT_MODEL,
    SPENDING_APPROVAL_MODEL,
)
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn

ERROR_AGENT_TOOL = "ui_exception_handler"


def _report(success: bool, **overrides: Any) -> ToolCall:
    """The error agent's structured report, worded by the (scripted) LLM."""
    return ToolCall(
        "UiExceptionReport",
        {
            "result": "done",
            "success": success,
            "process_finished": False,
            "continue_from_model_event_id": None,
            "loop_iteration": [],
        }
        | overrides,
    )


def _gateway_report(success: bool, **overrides: Any) -> ToolCall:
    return ToolCall(
        "ResponseToRPA",
        {
            "success": success,
            "process_finished": False,
            "continue_from_model_event_id": None,
            "loop_iteration": [],
        }
        | overrides,
    )


def _continuation(
    model_event_id: str | None,
    loop_iteration: list[int],
    process_finished: bool = False,
) -> ToolCall:
    return ToolCall(
        "set_continuation",
        {
            "process_finished": process_finished,
            "model_event_id": model_event_id,
            "loop_iteration": loop_iteration,
        },
    )


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-answer-0001"
    _ = make_robot_key(name="Answer", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        recovery_module, "tracer", provider.get_tracer("tests.recovery")
    )
    try:
        yield exporter
    finally:
        provider.shutdown()


@pytest.fixture
def agents(session: Session, mock_router: Router) -> tuple[Agent, Agent]:
    """A gateway whose only sub-agent is an error agent with `set_continuation`."""
    tool = Tool(
        id=uuid.uuid4(),
        name="set_continuation",
        description="Set where the robot continues.",
        fn_module="agent_tools.utils.set_continuation",
    )
    error_agent = Agent(
        id=uuid.uuid4(),
        name="UI Exception Handler",
        description="Recovers UI errors.",
        prompt="Recover the robot.",
        response_model="templates.error_handler.UiExceptionReport",
        input_type=Agent.InputType.TEXT,
        router=mock_router,
        router_id=mock_router.id,
        type=AgentType.ErrorAgent,
        arguments=[
            Argument(
                name="task_name",
                description="Task name.",
                type="str",
                json_type="string",
            )
        ],
    )
    gateway = Agent(
        id=uuid.uuid4(),
        name="Gateway Orchestrator",
        description="Routes recoveries.",
        prompt="Route the recovery.",
        response_model="templates.gateway.ResponseToRPA",
        input_type=Agent.InputType.TEXT,
        router=mock_router,
        router_id=mock_router.id,
        type=AgentType.GatewayAgent,
        arguments=[
            Argument(
                name="task_name",
                description="Task name.",
                type="str",
                json_type="string",
            )
        ],
    )
    session.add_all([tool, error_agent, gateway])
    session.commit()
    session.add_all(
        [
            AgentTool(
                agent_id=error_agent.id, tool_id=tool.id, limit=None, required=True
            ),
            SubAgent(
                parent_agent_id=gateway.id, child_agent_id=error_agent.id, limit=None
            ),
        ]
    )
    session.commit()
    session.refresh(gateway)
    session.refresh(error_agent)
    return gateway, error_agent


@pytest.fixture
def script(
    agents: tuple[Agent, Agent], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, ScriptedModel]]:
    """Give each agent a scripted model; the gateway's argument signature is free."""
    gateway, error_agent = agents

    def _script(
        *, gateway_turns: list[Turn], error_turns: list[Turn]
    ) -> dict[str, ScriptedModel]:
        models = {
            gateway.name: ScriptedModel(*gateway_turns),
            error_agent.name: ScriptedModel(*error_turns),
        }
        monkeypatch.setattr(Agent, "get_model", lambda self: models[self.name])  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        # Agent traces are written through the production engine; not under test.
        monkeypatch.setattr(Agent, "get_logging_hooks", lambda self, state: [])  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(Router, "get_conversation_cost", lambda self, i, o: 0.0)  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(Agent, "validate_input", lambda self, *a, **k: None)  # pyright: ignore[reportUnknownLambdaType]
        return models

    return _script


def _recover(
    client: TestClient, robot_key_raw: str, payload: dict[str, Any]
) -> dict[str, Any]:
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(payload)
        while True:
            message = ws.receive_json()
            if message["type"] != "ping":
                return message


def _turn(exporter: InMemorySpanExporter) -> ReadableSpan:
    return next(s for s in exporter.get_finished_spans() if s.name == "recovery.turn")


def _call_error_agent() -> ToolCall:
    return ToolCall(ERROR_AGENT_TOOL, {"task_name": "spending_approval"})


@pytest.fixture
def spending_payload(valid_recovery_payload: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = valid_recovery_payload | {
        "model": SPENDING_APPROVAL_MODEL
    }
    errored_act: dict[str, Any] = payload["errored_act"]
    payload["errored_act"] = errored_act | {
        "event_id": "8",
        "event_name": "Click Approve #2",
        "model_event_id": "click_approve",
    }
    return payload


def test_invalid_continuation_is_corrected_inside_the_agent_loop(
    session: Session,
    client: TestClient,
    robot_key_raw: str,
    spending_payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    span_exporter: InMemorySpanExporter,
):
    _ = script(
        gateway_turns=[
            _call_error_agent(),
            "Routed.",
            _gateway_report(True, continue_from_model_event_id="8"),
        ],
        error_turns=[
            # A UI log event_id, then a hallucinated task, then the valid one.
            _continuation("8", [2]),
            _continuation("click_aprove", [2]),
            _continuation("click_approve", [2]),
            "Recovered.",
            _report(True, continue_from_model_event_id="something_else"),
        ],
    )

    message = _recover(client, robot_key_raw, spending_payload)

    assert message["type"] == "done"
    assert message["content"] == {
        "success": True,
        "process_finished": False,
        "continue_from_model_event_id": "click_approve",
        "loop_iteration": [2],
        "variables": spending_payload["variables"],
    }
    exception = session.get(RobotException, uuid.UUID(message["id"]))
    assert exception is not None
    session.refresh(exception)
    assert exception.infered_success is True

    turn = _turn(span_exporter)
    assert turn.attributes is not None
    assert turn.attributes["recovery.continue_from_model_event_id"] == "click_approve"
    assert turn.attributes["recovery.loop_iteration"] == (2,)
    assert "recovery.failure_reason" not in turn.attributes


def test_success_without_continuation_is_sent_back_to_set_it(
    client: TestClient,
    robot_key_raw: str,
    spending_payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
):
    models = script(
        gateway_turns=[_call_error_agent(), "Routed.", _gateway_report(True)],
        error_turns=[
            "Recovered.",
            _continuation(None, [], process_finished=True),
            "Continuation set.",
            _report(True),
        ],
    )

    message = _recover(client, robot_key_raw, spending_payload)

    assert message["content"] == {
        "success": True,
        "process_finished": True,
        "continue_from_model_event_id": None,
        "loop_iteration": [],
        "variables": spending_payload["variables"],
    }
    prompts = [
        block.get("text")
        for msg in models["UI Exception Handler"].requests[-1]
        if msg["role"] == "user"
        for block in msg["content"]
    ]
    reminders = [p for p in prompts if p and "`set_continuation`" in p]
    assert len(reminders) == 1


def test_missing_continuation_is_sent_as_a_failure(
    session: Session,
    client: TestClient,
    robot_key_raw: str,
    spending_payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    span_exporter: InMemorySpanExporter,
):
    models = script(
        gateway_turns=[_call_error_agent(), "Routed.", _gateway_report(True)],
        error_turns=[
            "Recovered.",
            *["Still recovered."] * MAX_REQUIRED_TOOL_REMINDERS,
            _report(True),
        ],
    )

    message = _recover(client, robot_key_raw, spending_payload)

    assert message["content"] == {
        "success": False,
        "process_finished": False,
        "continue_from_model_event_id": None,
        "loop_iteration": [],
        "variables": spending_payload["variables"],
    }
    exception = session.get(RobotException, uuid.UUID(message["id"]))
    assert exception is not None
    session.refresh(exception)
    assert exception.infered_success is False

    turn = _turn(span_exporter)
    assert turn.attributes is not None
    assert turn.attributes["recovery.failure_reason"] == "missing_continuation"
    assert turn.attributes["recovery.success"] is False
    assert [e.name for e in turn.events].count("missing_continuation") == 1
    assert models["UI Exception Handler"].calls == 1 + MAX_REQUIRED_TOOL_REMINDERS + 1


def test_reported_failure_drops_a_stored_continuation(
    client: TestClient,
    robot_key_raw: str,
    spending_payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
):
    _ = script(
        gateway_turns=[_call_error_agent(), "Routed.", _gateway_report(False)],
        error_turns=[
            _continuation("click_approve", [2]),
            "Gave up.",
            _report(False),
        ],
    )

    message = _recover(client, robot_key_raw, spending_payload)

    assert message["content"]["success"] is False
    assert message["content"]["continue_from_model_event_id"] is None
    assert message["content"]["loop_iteration"] == []


def test_unchecked_loop_depth_is_sent_as_given(
    client: TestClient,
    robot_key_raw: str,
    valid_recovery_payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
):
    payload: dict[str, Any] = valid_recovery_payload | {"model": COMPACT_MODEL}
    _ = script(
        gateway_turns=[_call_error_agent(), "Routed.", _gateway_report(True)],
        error_turns=[
            _continuation("click_approve", [1, 2]),
            "Recovered.",
            _report(True),
        ],
    )

    message = _recover(client, robot_key_raw, payload)

    assert message["content"]["success"] is True
    assert message["content"]["continue_from_model_event_id"] == "click_approve"
    assert message["content"]["loop_iteration"] == [1, 2]


def test_variables_come_from_code_not_from_the_llm(
    session: Session,
    client: TestClient,
    robot_key_raw: str,
    valid_recovery_payload: dict[str, Any],
    gateway_agent: Agent,  # pyright: ignore[reportUnusedParameter]
    monkeypatch: pytest.MonkeyPatch,
):
    async def _fake_call(
        _self: Agent, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        assert invocation_state is not None
        invocation_state["variables"]["retry"] = 2
        kwargs["variables"]["invoice_id"] = "HALLUCINATED"
        return {"success": False, "variables": {"invented": True}}

    monkeypatch.setattr(Agent, "__call__", _fake_call)

    message = _recover(client, robot_key_raw, valid_recovery_payload)

    assert message["content"]["variables"] == {"invoice_id": "INV-100", "retry": 2}
    stored = session.exec(select(RobotException)).one()
    assert stored.exception_details["variables"] == {
        "invoice_id": "INV-100",
        "retry": 1,
    }
