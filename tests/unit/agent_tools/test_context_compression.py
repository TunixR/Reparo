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
    ContextCompression,
    ReportingSummarizingManager,
)
from tests.unit.shared.log_events import logged_events
from tests.unit.shared.scripted_model import ScriptedModel, Summary, ToolCall, Turn

PING = ToolCall("ping", {})
FULL = (10**9, 4)
"""Usage that puts every model call past any context window."""
SMALL = (10, 1)
"""Usage far below the compression threshold."""
OVERFLOW = ContextWindowOverflowException("prompt too long")


@tool
def ping() -> str:
    """Answer pong, at length: the summary is shorter than what it replaces."""
    return "pong " * 100


def _agent(
    *turns: Turn,
    usage: tuple[int, int] = FULL,
    summary: Summary = "Earlier steps, summarized.",
) -> tuple[StrandsAgent, ScriptedModel, list[ContextCompression]]:
    """An agent whose task message is pinned and whose last two messages are kept."""
    model = ScriptedModel(*turns, usage=usage, summary=summary)
    reports: list[ContextCompression] = []
    agent = StrandsAgent(
        model=model,
        tools=[ping],
        conversation_manager=ReportingSummarizingManager(
            pin_first=1, preserve_recent_messages=2, on_compression=reports.append
        ),
        callback_handler=None,
    )
    return agent, model, reports


def _text(message: Any) -> str:
    return "".join(block.get("text", "") for block in message["content"])


@pytest.mark.asyncio
async def test_a_full_context_is_summarized_and_reported() -> None:
    agent, model, reports = _agent(PING, PING, PING, PING, "done")

    _ = await agent.invoke_async("Fill the invoice")

    first = reports[0]
    assert first.trigger == "proactive"
    assert first.summary == "Earlier steps, summarized."
    # The third call: the task is pinned, the first ping call and its result go.
    assert first.messages_replaced == 2
    assert first.tokens_before is not None and first.tokens_before > FULL[0]
    assert first.tokens_after is not None and first.tokens_after < first.tokens_before
    assert (first.input_tokens, first.output_tokens) == (50, 5)
    # Every request still starts with the task, and the summary follows it.
    assert all(_text(request[0]) == "Fill the invoice" for request in model.requests)
    assert _text(model.requests[2][1]) == "Earlier steps, summarized."


@pytest.mark.asyncio
async def test_an_overflow_is_summarized_and_reported() -> None:
    # Small usage: only the overflow makes the agent reduce its context.
    agent, model, reports = _agent(PING, PING, PING, OVERFLOW, "done", usage=SMALL)

    _ = await agent.invoke_async("Fill the invoice")

    [report] = reports
    assert report.trigger == "overflow"
    assert report.messages_replaced == 2
    assert _text(model.requests[-1][1]) == "Earlier steps, summarized."


@pytest.mark.asyncio
async def test_a_failed_summary_is_logged_and_the_run_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    agent, model, reports = _agent(
        PING, PING, PING, "done", summary=RuntimeError("summarizer down")
    )

    result = await agent.invoke_async("Fill the invoice")

    assert _text(result.message) == "done"
    assert reports == []
    # Nothing was replaced: the task and the three ping turns reach the last call.
    assert len(model.requests[-1]) == 1 + 2 * 3
    failures = logged_events(caplog, "agent.context_compression_failed")
    assert {vars(record)["error.type"] for record in failures} >= {"RuntimeError"}
    assert all(record.levelno == logging.WARNING for record in failures)


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
    # Not a recovered failure: the run's own error records it.
    assert logged_events(caplog, "agent.context_compression_failed") == []


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
async def test_a_summary_is_a_span_event(exporter: InMemorySpanExporter) -> None:
    agent, _, reports = _agent(PING, PING, PING, "done")

    _ = await agent.invoke_async("Fill the invoice")

    events = [
        event
        for span in exporter.get_finished_spans()
        for event in span.events
        if event.name == "agent.context_compressed"
    ]
    assert len(events) == len(reports) > 0
    first = dict(events[0].attributes or {})
    before = first.pop("agent.context_compression.tokens_before")
    after = first.pop("agent.context_compression.tokens_after")
    assert isinstance(before, int) and isinstance(after, int)
    assert before > FULL[0] and after < before
    assert first == {
        "agent.context_compression.trigger": "proactive",
        "agent.context_compression.messages_replaced": 2,
        "gen_ai.usage.input_tokens": 50,
        "gen_ai.usage.output_tokens": 5,
    }
