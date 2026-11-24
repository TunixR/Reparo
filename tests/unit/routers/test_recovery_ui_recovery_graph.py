"""A recovery runs the seeded "UI Recovery" graph end to end over the WebSocket.

The agents and the graph come from the real populators. Each agent runs a real
Strands agent over a scripted model, and the GUI agent drives the test's robot
through screenshot and action frames.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import Engine
from sqlmodel import Session, select
from strands.telemetry import tracer as strands_tracer

import observability.instrumentation.agent_graph as agent_graph_telemetry
import observability.instrumentation.gui_agent as gui_agent_telemetry
import routers.recovery as recovery_module
from database.agent_graphs.models import AgentGraph
from database.agent_graphs.runner import NodeRun
from database.agents.behaviours import AgentBehaviour, DefaultAgent
from database.agents.models import Agent
from database.populators.agent_graphs import populate_agent_graphs
from database.provider.models import Router
from tests.unit.agent_tools._shared.process_models import SPENDING_APPROVAL_MODEL
from tests.unit.shared.fake_websocket import make_test_image_bytes
from tests.unit.shared.gui_fakes import patch_gui_io
from tests.unit.shared.log_events import logged_events
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn

GATEWAY = "Gateway Orchestrator"
INFERER = "Recovery Inferer"
GUI = "Tooled GuiAgent"
CONTINUATION = "Continuation Inferer"


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-ui-recovery-01"
    _ = make_robot_key(name="UI Recovery", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def payload(valid_recovery_payload: dict[str, Any]) -> dict[str, Any]:
    return valid_recovery_payload | {"model": SPENDING_APPROVAL_MODEL}


@pytest.fixture
def ui_recovery(engine: Engine, seeded_agents: dict[str, Agent]) -> None:
    populate_agent_graphs(engine)


@pytest.fixture
def script(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., dict[str, ScriptedModel]]:
    """Give each seeded agent, by name, a scripted model."""

    def _script(turns: Mapping[str, list[Turn]]) -> dict[str, ScriptedModel]:
        models = {
            name: ScriptedModel(*turns.get(name, ["Unused."]))
            for name in (GATEWAY, INFERER, GUI, CONTINUATION)
        }
        monkeypatch.setattr(Agent, "get_model", lambda self: models[self.name])  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        monkeypatch.setattr(DefaultAgent, "logging_hooks", lambda self, state: [])  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(Router, "get_conversation_cost", lambda self, i, o: 0.0)  # pyright: ignore[reportUnknownLambdaType]
        patch_gui_io(monkeypatch)
        return models

    return _script


@pytest.fixture
def node_runs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The agent of each node run, in order."""
    runs: list[str] = []
    original = AgentBehaviour.run_node

    async def _spy(
        self: AgentBehaviour,
        invocation_state: dict[str, Any],
        input_map: Mapping[str, str],
    ) -> NodeRun:
        runs.append(self.agent.name)
        return await original(self, invocation_state, input_map)

    monkeypatch.setattr(AgentBehaviour, "run_node", _spy)
    return runs


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Export the recovery's, the graph's, the GUI agent's and Strands' spans."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tests.ui_recovery")
    for module in (recovery_module, agent_graph_telemetry, gui_agent_telemetry):
        monkeypatch.setattr(module, "tracer", tracer)
    monkeypatch.setattr(strands_tracer.get_tracer(), "tracer", tracer)
    try:
        yield exporter
    finally:
        provider.shutdown()


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _children(
    exporter: InMemorySpanExporter, parent: ReadableSpan
) -> list[ReadableSpan]:
    assert parent.context is not None
    return [
        span
        for span in exporter.get_finished_spans()
        if span.parent is not None and span.parent.span_id == parent.context.span_id
    ]


def _recover(
    client: TestClient, robot_key_raw: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Play the robot: the answer, and the actions it was asked to perform."""
    actions: list[dict[str, Any]] = []
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(payload)
        while True:
            message = ws.receive_json()
            match message["type"]:
                case "ping":
                    continue
                case "screenshot":
                    ws.send_bytes(make_test_image_bytes(size=(64, 48)))
                case "action":
                    actions.append(message)
                    ws.send_json({"success": True})
                case _:
                    return message, actions


def _texts(model: ScriptedModel) -> str:
    """Every user text a model was sent."""
    return "\n".join(
        block["text"]
        for request in model.requests
        for message in request
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    )


def _classified(*, ui_exception: bool) -> list[Turn]:
    return [
        "Classified.",
        ToolCall(
            "GatewayDecision",
            {"ui_exception": ui_exception, "rationale": "The Approve click failed."},
        ),
    ]


INFERRED: list[Turn] = [
    "Inferred.",
    ToolCall(
        "RecoveryInference",
        {
            "failure_kind": "navigation",
            "failed_model_event_id": "click_approve",
            "loop_passes": [2],
            "recovery_task": "Approve the current spending request",
            "rationale": "A later pass, so only the current request.",
        },
    ),
]
RECOVERED_ON_THE_DESKTOP: list[Turn] = [
    ToolCall("click", {"start_box": [500, 500]}),
    ToolCall("finished", {"content": "Request approved"}),
]
CONTINUED: list[Turn] = [
    ToolCall(
        "set_continuation",
        {
            "recovered": True,
            "process_finished": False,
            "model_event_id": "back_to_list",
            "loop_iteration": [2],
        },
    ),
    "Continuation set.",
    ToolCall(
        "ContinuationReport",
        {"success": True, "summary": "Approved; the robot returns to the list."},
    ),
]


@pytest.mark.usefixtures("ui_recovery")
def test_a_ui_failure_runs_all_four_nodes_and_is_recovered(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[str],
):
    models = script(
        {
            GATEWAY: _classified(ui_exception=True),
            INFERER: INFERRED,
            GUI: RECOVERED_ON_THE_DESKTOP,
            CONTINUATION: CONTINUED,
        }
    )

    message, actions = _recover(client, robot_key_raw, payload)

    assert node_runs == [GATEWAY, INFERER, GUI, CONTINUATION]
    assert message["type"] == "done"
    assert message["content"] == {
        "success": True,
        "process_finished": False,
        "continue_from_model_event_id": "back_to_list",
        "loop_iteration": [2],
        "variables": payload["variables"],
    }
    assert len(actions) == 1
    assert "Task: Approve the current spending request" in _texts(models[GUI])
    continuation_inputs = _texts(models[CONTINUATION])
    assert "recovery_task: Approve the current spending request" in continuation_inputs
    assert "gui_summary: Request approved" in continuation_inputs
    assert "goal_reached: True" in continuation_inputs


@pytest.mark.usefixtures("ui_recovery")
def test_a_failure_that_is_no_ui_exception_runs_only_the_gateway(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[str],
):
    models = script({GATEWAY: _classified(ui_exception=False)})

    message, actions = _recover(client, robot_key_raw, payload)

    assert node_runs == [GATEWAY]
    assert message["type"] == "done"
    assert message["content"]["success"] is False
    assert actions == []
    assert all(models[name].calls == 0 for name in (INFERER, GUI, CONTINUATION))


@pytest.mark.usefixtures("ui_recovery")
def test_the_run_and_each_node_are_spans_between_the_turn_and_the_agents(
    client: TestClient,
    session: Session,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    spans: InMemorySpanExporter,
):
    _ = script(
        {
            GATEWAY: _classified(ui_exception=True),
            INFERER: INFERRED,
            GUI: RECOVERED_ON_THE_DESKTOP,
            CONTINUATION: CONTINUED,
        }
    )
    graph = session.exec(select(AgentGraph)).one()
    agent_ids = {node.key: str(node.agent_id) for node in graph.nodes}

    message, _ = _recover(client, robot_key_raw, payload)

    assert message["type"] == "done"
    [turn] = _spans(spans, "recovery.turn")
    [run] = [s for s in _children(spans, turn) if s.name == "agent_graph.run"]
    assert dict(run.attributes or {}) == {
        "agent_graph.id": str(graph.id),
        "agent_graph.name": "UI Recovery",
        "agent_graph.outcome": "completed",
        "agent_graph.terminal_node.key": "continuation_inferer",
        "agent_graph.nodes_run": 4,
    }
    nodes = _children(spans, run)
    assert [
        (
            (n.attributes or {})["agent_graph.node.key"],
            (n.attributes or {})["agent.id"],
            (n.attributes or {})["agent_graph.node.execution_index"],
        )
        for n in sorted(nodes, key=lambda n: n.start_time or 0)
    ] == [
        ("gateway", agent_ids["gateway"], 0),
        ("recovery_inferer", agent_ids["recovery_inferer"], 1),
        ("tooled_guiagent", agent_ids["tooled_guiagent"], 2),
        ("continuation_inferer", agent_ids["continuation_inferer"], 3),
    ]
    assert {n.name for n in nodes} == {"agent_graph.node"}
    for node in nodes:
        assert any(s.name.startswith("invoke_agent") for s in _children(spans, node))


@pytest.mark.usefixtures("ui_recovery")
def test_the_gui_node_span_describes_the_gui_task(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    spans: InMemorySpanExporter,
):
    _ = script(
        {
            GATEWAY: _classified(ui_exception=True),
            INFERER: INFERRED,
            GUI: RECOVERED_ON_THE_DESKTOP,
            CONTINUATION: CONTINUED,
        }
    )

    _ = _recover(client, robot_key_raw, payload)

    [gui_node] = [
        span
        for span in _spans(spans, "agent_graph.node")
        if (span.attributes or {})["agent_graph.node.key"] == "tooled_guiagent"
    ]
    attributes = gui_node.attributes or {}
    assert (
        attributes["gui.recovery_task.name"] == "Approve the current spending request"
    )
    assert attributes["gui.task.outcome"] == "done"
    assert attributes["gui.task.iterations"] == 1
    assert attributes["gui.task.ui_log_count"] == len(payload["ui_log"])
    assert any(s.name == "gui.screenshot" for s in _children(spans, gui_node))


def _graph_run(spans: InMemorySpanExporter) -> dict[str, Any]:
    [run] = _spans(spans, "agent_graph.run")
    return dict(run.attributes or {})


@pytest.mark.usefixtures("ui_recovery")
def test_a_run_that_hits_its_node_limit_is_failed(
    client: TestClient,
    session: Session,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    spans: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
):
    graph = session.exec(select(AgentGraph)).one()
    graph.max_node_executions = 2
    session.add(graph)
    session.commit()
    _ = script({GATEWAY: _classified(ui_exception=True), INFERER: INFERRED})

    message, _ = _recover(client, robot_key_raw, payload)

    assert message["content"]["success"] is False
    run = _graph_run(spans)
    assert run["agent_graph.outcome"] == "max_node_executions"
    assert run["agent_graph.nodes_run"] == 2
    assert run["agent_graph.terminal_node.key"] == "recovery_inferer"
    [failed] = logged_events(caplog, "agent_graph.failed")
    assert vars(failed)["agent_graph.id"] == str(graph.id)
    assert vars(failed)["agent_graph.outcome"] == "max_node_executions"


@pytest.mark.usefixtures("ui_recovery")
def test_a_node_that_raises_fails_the_run_and_its_node(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    spans: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
):
    _ = script(
        {GATEWAY: _classified(ui_exception=True), INFERER: [RuntimeError("down")]}
    )

    message, _ = _recover(client, robot_key_raw, payload)

    assert message["code"] == "INTERNAL_ERROR"
    run = _graph_run(spans)
    assert run["agent_graph.outcome"] == "error"
    assert run["agent_graph.nodes_run"] == 1
    assert run["agent_graph.terminal_node.key"] == "gateway"
    [inferer] = [
        span
        for span in _spans(spans, "agent_graph.node")
        if (span.attributes or {})["agent_graph.node.key"] == "recovery_inferer"
    ]
    assert inferer.status.status_code.name == "ERROR"
    assert (inferer.attributes or {})["error.type"] == "RuntimeError"
    [failed] = logged_events(caplog, "agent_graph.failed")
    assert vars(failed)["agent_graph.outcome"] == "error"
    assert vars(failed)["error.type"] == "RuntimeError"


@pytest.mark.usefixtures("ui_recovery")
def test_a_failing_gui_node_records_its_exception_once(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    spans: InMemorySpanExporter,
):
    _ = script(
        {
            GATEWAY: _classified(ui_exception=True),
            INFERER: INFERRED,
            GUI: [ValueError("model down")],
        }
    )

    message, _ = _recover(client, robot_key_raw, payload)

    assert message["code"] == "INTERNAL_ERROR"
    [gui_node] = [
        span
        for span in _spans(spans, "agent_graph.node")
        if (span.attributes or {})["agent_graph.node.key"] == "tooled_guiagent"
    ]
    assert (gui_node.attributes or {})["gui.task.outcome"] == "error"
    assert [event.name for event in gui_node.events] == ["exception"]
