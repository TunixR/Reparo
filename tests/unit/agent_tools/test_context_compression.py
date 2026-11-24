"""The recovery conversation manager summarizes old context and reports each summary."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent as StrandsAgent
from strands import tool
from strands.telemetry import tracer as strands_tracer
from strands.types.exceptions import (
    ContextWindowOverflowException,
    EventLoopException,
)

from agent_tools.context_compression import (
    OMITTED_IMAGE_TEXT,
    ContextCompression,
    RecoveryConversationManager,
)
from tests.unit.shared.log_events import logged_events
from tests.unit.shared.scripted_model import ScriptedModel, Summary, ToolCall, Turn

PING = ToolCall("ping", {})
WINDOW = 1_000
FULL = (800, 4)
"""Usage past 70% of `WINDOW`: every later model call is due for a summary."""
SMALL = (10, 1)
"""Usage far below the compression threshold."""
OVERFLOW = ContextWindowOverflowException("prompt too long")


@tool
def ping() -> str:
    """Answer pong, at length: the summary is shorter than what it replaces."""
    return "pong " * 100


@tool
def look() -> dict[str, Any]:
    """Show the screen."""
    return {
        "status": "success",
        "content": [{"image": {"format": "jpeg", "source": {"bytes": b"jpeg"}}}],
    }


def _agent(
    *turns: Turn,
    usage: tuple[int, int] = FULL,
    summary: Summary = "Earlier steps, summarized.",
) -> tuple[StrandsAgent, ScriptedModel, list[ContextCompression]]:
    """An agent whose task message is pinned and whose last two messages are kept."""
    model = ScriptedModel(
        *turns, usage=usage, context_window_limit=WINDOW, summary=summary
    )
    reports: list[ContextCompression] = []
    agent = StrandsAgent(
        model=model,
        tools=[ping, look, dump],
        conversation_manager=RecoveryConversationManager(
            pin_first=1, preserve_recent_messages=2, on_compression=reports.append
        ),
        callback_handler=None,
    )
    return agent, model, reports


def _text(message: Any) -> str:
    return "".join(block.get("text", "") for block in message["content"])


@pytest.mark.asyncio
async def test_a_full_context_is_summarized_after_the_pinned_messages() -> None:
    agent, model, reports = _agent(PING, PING, PING, PING, "done")

    _ = await agent.invoke_async("Fill the invoice")

    first = reports[0]
    assert first.trigger == "proactive"
    assert first.summary == "Earlier steps, summarized."
    # The third call: the task is pinned, the first ping call and its result go.
    assert first.messages_replaced == 2
    assert first.tokens_before is not None and first.tokens_before >= FULL[0]
    assert first.tokens_after is not None and first.tokens_after < first.tokens_before
    assert (first.input_tokens, first.output_tokens) == (50, 5)
    assert all(_text(request[0]) == "Fill the invoice" for request in model.requests)
    assert _text(model.requests[2][1]) == "Earlier steps, summarized."


@pytest.mark.asyncio
async def test_an_overflow_is_summarized_and_the_call_retried() -> None:
    agent, model, reports = _agent(PING, PING, PING, OVERFLOW, "done", usage=SMALL)

    result = await agent.invoke_async("Fill the invoice")

    assert _text(result.message) == "done"
    [report] = reports
    assert report.trigger == "overflow"
    assert report.messages_replaced == 2
    assert report.tokens_before is None and report.tokens_after is None
    assert _text(model.requests[-1][1]) == "Earlier steps, summarized."


@pytest.mark.parametrize(
    ("turns", "usage"),
    [
        pytest.param((PING, PING, PING, "done"), FULL, id="proactive"),
        pytest.param((PING, PING, PING, OVERFLOW, "done"), SMALL, id="overflow"),
    ],
)
@pytest.mark.asyncio
async def test_the_summary_call_does_not_block_the_event_loop(
    turns: tuple[Turn, ...], usage: tuple[int, int]
) -> None:
    loop = asyncio.get_running_loop()

    def summary() -> str:
        # Runs while the agent waits for its summary: the loop must still serve others.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)
        return "Summarized while the loop ran."

    agent, _, reports = _agent(*turns, usage=usage, summary=summary)

    _ = await agent.invoke_async("Fill the invoice")

    assert reports and reports[0].summary == "Summarized while the loop ran."


@pytest.mark.asyncio
async def test_a_failed_summary_is_a_warning_and_the_run_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    agent, model, reports = _agent(
        PING, PING, PING, "done", summary=RuntimeError("summarizer down")
    )

    result = await agent.invoke_async("Fill the invoice")

    assert _text(result.message) == "done"
    assert reports == []
    assert len(model.requests[-1]) == 1 + 2 * 3
    failures = logged_events(caplog, "agent.context_compression_failed")
    assert failures
    assert all(record.levelno == logging.WARNING for record in failures)
    assert "RuntimeError" in {vars(record)["error.type"] for record in failures}
    assert {
        vars(record)["agent.context_compression.trigger"] for record in failures
    } == {"proactive"}


@tool
def dump() -> str:
    """Dump the page source."""
    return "x" * 10_000


@pytest.mark.asyncio
async def test_a_summary_that_overflows_too_truncates_the_largest_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    dump_ = ToolCall("dump", {})
    agent, model, reports = _agent(
        dump_,
        dump_,
        dump_,
        OVERFLOW,
        "done",
        usage=SMALL,
        summary=ContextWindowOverflowException("summary too long"),
    )

    result = await agent.invoke_async("Fill the invoice")

    assert _text(result.message) == "done"
    assert reports == []
    retried = str(model.requests[-1])
    assert retried.count("x" * 10_000) == 2
    [failure] = [
        record
        for record in logged_events(caplog, "agent.context_compression_failed")
        if vars(record)["agent.context_compression.trigger"] == "overflow"
    ]
    assert vars(failure)["error.type"] == "ContextWindowOverflowException"


@pytest.mark.asyncio
async def test_a_failed_summary_after_an_overflow_ends_the_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    agent, _, reports = _agent(
        PING, PING, PING, OVERFLOW, usage=SMALL, summary=RuntimeError("summarizer down")
    )

    with pytest.raises(EventLoopException, match="summarizer down"):
        _ = await agent.invoke_async("Fill the invoice")

    assert reports == []
    assert logged_events(caplog, "agent.context_compression_failed") == []


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    """Route the Strands spans to an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        strands_tracer.get_tracer(), "tracer", provider.get_tracer("tests")
    )
    yield exporter
    provider.shutdown()


@pytest.mark.asyncio
async def test_a_summary_is_an_event_on_the_cycle_span(
    exporter: InMemorySpanExporter,
) -> None:
    agent, _, reports = _agent(PING, PING, PING, "done")

    _ = await agent.invoke_async("Fill the invoice")

    events = [
        (span.name, event)
        for span in exporter.get_finished_spans()
        for event in span.events
        if event.name == "agent.context_compressed"
    ]
    assert len(events) == len(reports) > 0
    span_name, event = events[0]
    assert span_name == "execute_event_loop_cycle"
    attributes = dict(event.attributes or {})
    before = attributes.pop("agent.context_compression.tokens_before")
    after = attributes.pop("agent.context_compression.tokens_after")
    assert isinstance(before, int) and isinstance(after, int) and after < before
    assert attributes == {
        "agent.context_compression.trigger": "proactive",
        "agent.context_compression.messages_replaced": 2,
        "gen_ai.usage.input_tokens": 50,
        "gen_ai.usage.output_tokens": 5,
    }


def _images(messages: list[Any]) -> int:
    return sum(
        "image" in block
        or any(
            "image" in item for item in block.get("toolResult", {}).get("content", [])
        )
        for message in messages
        for block in message["content"]
    )


@pytest.mark.asyncio
async def test_the_summarizer_gets_placeholders_for_the_screenshots() -> None:
    look_ = ToolCall("look", {})
    agent, model, reports = _agent(look_, look_, look_, "done")

    _ = await agent.invoke_async("Fill the invoice")

    assert reports and model.summary_requests
    summarized = [m for request in model.summary_requests for m in request]
    assert _images(summarized) == 0
    assert any(OMITTED_IMAGE_TEXT in str(m["content"]) for m in summarized)
    # The kept steps still show their screenshots to the agent.
    assert _images(model.requests[-1]) > 0
