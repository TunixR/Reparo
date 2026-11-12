"""The eval's answer: the production contract, from the same continuation tool.

The GUI agent is faked; the continuation inferrer is a real Strands agent over a
scripted model, so `set_continuation`, its required-tool reminders and the
invocation state run as they do for the production error agent.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import routers.evals as evals_module
from agent_tools.hooks import MAX_REQUIRED_TOOL_REMINDERS
from tests.unit.agent_tools._shared.process_models import (
    COMPACT_MODEL,
    SPENDING_APPROVAL_MODEL,
)
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn

VARIABLES = {"pending": ["r1", "r2", "r3", "r4"], "request_id": "r3", "amount": 120}


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


def _report(success: bool, **overrides: Any) -> ToolCall:
    """The inferrer's structured report, worded by the (scripted) LLM."""
    return ToolCall("ContinuationOutput", {"success": success} | overrides)


class _FakeGuiAgent:
    """Stands in for a GUI agent row: its tool changes one variable, like `alter_variable`."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def as_tool(self):
        async def run(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            self.calls.append(kwargs)
            variables = kwargs["override_invocation_state"]["variables"]
            variables["request_id"] = "r4"
            return "clicked Approve", variables

        return run


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


def _run(
    monkeypatch: pytest.MonkeyPatch,
    exporter: InMemorySpanExporter,
    turns: list[Turn],
    model: str = SPENDING_APPROVAL_MODEL,
) -> tuple[dict[str, Any], ScriptedModel, ReadableSpan]:
    scripted = ScriptedModel(*turns)
    monkeypatch.setattr(evals_module, "TracedOpenAIModel", lambda **_: scripted)
    payload = {
        "variables": VARIABLES,
        "ui_log": [],
        "errored_act": {"event_id": "8", "model_event_id": "click_approve"},
        "model": model,
    }

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("tests.evals").start_as_current_span("recovery.eval"):
        result = asyncio.run(
            evals_module._run_standalone_recovery(  # pyright: ignore[reportPrivateUsage]
                channel=object(),  # pyright: ignore[reportArgumentType]
                payload=payload,
                recovery_task="Approve the request",
                agent=_FakeGuiAgent(),  # pyright: ignore[reportArgumentType]
                robot_exception_id="exc-1",
            )
        )
    provider.shutdown()
    (span,) = exporter.get_finished_spans()
    return result, scripted, span


def test_answer_has_the_production_shape_and_the_validated_continuation(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
):
    result, _, span = _run(
        monkeypatch,
        exporter,
        [
            # A UI log event_id, a wrong depth, then the valid continuation.
            _continuation("8", [2]),
            _continuation("back_to_list", []),
            _continuation("back_to_list", [2]),
            "Continuation set.",
            # The LLM's wording can't change the stored continuation.
            _report(True, continue_from_model_event_id="click_reject"),
        ],
    )

    assert result == {
        "success": True,
        "process_finished": False,
        "continue_from_model_event_id": "back_to_list",
        "loop_iteration": [2],
        # From code: the payload's variables as the GUI agent left them.
        "variables": VARIABLES | {"request_id": "r4"},
    }
    assert span.attributes is not None
    assert span.attributes["recovery.success"] is True
    assert span.attributes["recovery.continue_from_model_event_id"] == "back_to_list"
    assert span.attributes["recovery.loop_iteration"] == (2,)
    assert "recovery.failure_reason" not in span.attributes


def test_payload_variables_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
):
    _ = _run(
        monkeypatch,
        exporter,
        [_continuation(None, [], process_finished=True), "Done.", _report(True)],
    )

    assert VARIABLES["request_id"] == "r3"


def test_success_without_continuation_is_sent_back_to_set_it(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
):
    result, scripted, _ = _run(
        monkeypatch,
        exporter,
        [
            "Recovered.",
            _continuation(None, [], process_finished=True),
            "Continuation set.",
            _report(True),
        ],
    )

    assert result["success"] is True
    assert result["process_finished"] is True
    assert result["continue_from_model_event_id"] is None
    assert result["loop_iteration"] == []
    prompts = [
        block.get("text")
        for msg in scripted.requests[-1]
        if msg["role"] == "user"
        for block in msg["content"]
    ]
    assert len([p for p in prompts if p and "`set_continuation`" in p]) == 1


def test_missing_continuation_is_scored_as_a_failure(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
):
    result, scripted, span = _run(
        monkeypatch,
        exporter,
        [
            "Recovered.",
            *["Still recovered."] * MAX_REQUIRED_TOOL_REMINDERS,
            _report(True),
        ],
    )

    assert result["success"] is False
    assert result["continue_from_model_event_id"] is None
    assert result["loop_iteration"] == []
    assert scripted.calls == 1 + MAX_REQUIRED_TOOL_REMINDERS + 1
    assert span.attributes is not None
    assert span.attributes["recovery.failure_reason"] == "missing_continuation"
    assert [e.name for e in span.events] == ["missing_continuation"]


def test_reported_failure_drops_a_stored_continuation(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
):
    result, _, _ = _run(
        monkeypatch,
        exporter,
        [_continuation("click_approve", [2]), "Not fulfilled.", _report(False)],
    )

    assert result["success"] is False
    assert result["process_finished"] is False
    assert result["continue_from_model_event_id"] is None
    assert result["loop_iteration"] == []


def test_depth_is_not_checked_on_a_textual_model(
    monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
):
    result, _, _ = _run(
        monkeypatch,
        exporter,
        [_continuation("click_approve", [1, 2]), "Set.", _report(True)],
        model=COMPACT_MODEL,
    )

    assert result["continue_from_model_event_id"] == "click_approve"
    assert result["loop_iteration"] == [1, 2]


def test_structured_report_only_exposes_success():
    schema = evals_module.ContinuationOutput.model_json_schema()

    assert set(schema["properties"]) == {"success"}
