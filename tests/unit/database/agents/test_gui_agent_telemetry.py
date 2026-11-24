"""A GUI agent reports its task, actions and screenshots through GuiTaskTelemetry."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import WebSocketDisconnect
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

import database.agents.behaviours.gui.base as gui_base
import observability.instrumentation.gui_agent as gui_agent_module
from database.agents.behaviours.gui import GuiAgent
from database.agents.behaviours.gui.shared import GuiRun
from tests.unit.shared.fake_websocket import FakeWebSocket
from tests.unit.shared.gui_fakes import (
    GUI_KINDS,
    clicking_gui_agent,
    drive,
    make_gui_agent,
    make_robot,
    patch_gui_io,
)
from tests.unit.shared.scripted_model import ToolCall

pytestmark = pytest.mark.asyncio

TYPED_SECRET = "typed-secret-value"


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Tracer]:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tests.tooled_gui")
    monkeypatch.setattr(gui_agent_module, "tracer", tracer)
    yield tracer
    provider.shutdown()


@pytest.fixture(autouse=True)
def _no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_gui_io(monkeypatch)


async def _run(
    tracer: Tracer, gui_agent: GuiAgent, robot: FakeWebSocket
) -> tuple[GuiRun, int]:
    """Run the agent inside a stand-in for its Strands ``execute_tool`` span."""
    with tracer.start_as_current_span("execute_tool test_gui_agent"):
        run = await drive(gui_agent, robot)
    return run, len(robot.actions)


def _spans(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _task_attributes(exporter: InMemorySpanExporter) -> dict[str, Any]:
    [task] = _spans(exporter, "execute_tool test_gui_agent")
    return dict(task.attributes or {})


async def test_finished_task_reports_actions_and_screenshots(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    gui_agent = make_gui_agent(
        ToolCall("click", {"start_box": [10, 20]}),
        ToolCall("finished", {"content": "Invoice filled"}),
    )

    _, sent = await _run(tracer, gui_agent, make_robot())

    assert sent == 1
    task = _task_attributes(exporter)
    assert task["gui.recovery_task.name"] == "Fill the invoice"
    assert task["gui.task.process_model"] == "invoice-process"
    assert task["gui.task.ui_log_count"] == 2
    assert task["gui.task.has_errored_activity"] is True
    assert task["gui.task.outcome"] == "done"
    assert task["gui.task.iterations"] == 1

    [action] = _spans(exporter, "gui.action")
    assert action.attributes == {
        "gui.action.iteration": 1,
        "gui.action.type": "left_single",
        "gui.action.success": True,
    }
    # One before the loop, one after the click.
    screenshots = _spans(exporter, "gui.screenshot")
    assert len(screenshots) == 2
    assert screenshots[0].attributes is not None
    assert screenshots[0].attributes["gui.screenshot.width"] == 64
    assert screenshots[0].attributes["gui.screenshot.height"] == 48


async def test_failed_action_is_recorded_without_its_inputs(
    tracer: Tracer, exporter: InMemorySpanExporter
) -> None:
    gui_agent = make_gui_agent(
        ToolCall("type_text", {"content": TYPED_SECRET}), "Gave up."
    )

    await _run(tracer, gui_agent, make_robot(action_success=False))

    [action] = _spans(exporter, "gui.action")
    assert action.attributes is not None
    assert action.attributes["gui.action.success"] is False
    assert action.status.status_code.name == "ERROR"
    [event] = action.events
    assert event.name == "exception"
    assert TYPED_SECRET not in str(event.attributes)
    # Strands hands the failure back to the model, which may still go on.
    task = _task_attributes(exporter)
    assert task["gui.task.outcome"] == "model_stopped"
    assert task["gui.task.iterations"] == 1


async def test_unexpected_error_is_recorded_on_the_task_span(
    tracer: Tracer, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    gui_agent = make_gui_agent("unused")

    async def _broken(*_: Any, **__: Any) -> Any:
        raise ConnectionError("model endpoint unreachable")

    monkeypatch.setattr(gui_base.StrandsAgent, "invoke_async", _broken)

    run, _ = await _run(tracer, gui_agent, make_robot())

    assert run.error == "model endpoint unreachable"
    [task] = _spans(exporter, "execute_tool test_gui_agent")
    assert (task.attributes or {})["gui.task.outcome"] == "error"
    assert task.status.status_code.name == "ERROR"
    assert [event.name for event in task.events] == ["exception"]


@pytest.mark.parametrize("kind", GUI_KINDS)
async def test_a_robot_disconnect_ends_either_loop(
    tracer: Tracer,
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[GuiAgent],
) -> None:
    robot = make_robot()

    async def _gone() -> dict[str, Any]:
        raise WebSocketDisconnect(code=1006)

    monkeypatch.setattr(robot, "receive_json", _gone)

    with pytest.raises(WebSocketDisconnect):
        _ = await _run(tracer, clicking_gui_agent(kind), robot)

    assert len(robot.actions) == 1
    assert _task_attributes(exporter)["gui.task.outcome"] == "client_disconnected"


@pytest.mark.parametrize("kind", GUI_KINDS)
async def test_a_robot_abort_ends_either_loop(
    tracer: Tracer,
    exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    kind: type[GuiAgent],
) -> None:
    robot = make_robot()
    waiting = asyncio.Event()

    async def _never_answers() -> dict[str, Any]:
        waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(robot, "receive_json", _never_answers)
    run = asyncio.create_task(_run(tracer, clicking_gui_agent(kind), robot))
    _ = await waiting.wait()

    # The robot's abort cancels the recovery, mid-action.
    _ = run.cancel()
    with pytest.raises(asyncio.CancelledError):
        _ = await run

    assert len(robot.actions) == 1
    assert _task_attributes(exporter)["gui.task.outcome"] == "cancelled"
