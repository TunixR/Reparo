"""Token accounting of an agent run across its two Strands invocations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from strands import Agent as StrandsAgent

from agent_tools.hooks import AgentLoggingHook
from database.provider.models import Router
from tests.unit.fixtures.agent_fixtures import EntryPoint, ScriptedAgent
from tests.unit.shared.gui_fakes import FakeHook, fake_router, make_gui_agent
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn


def _usage_event(caplog: pytest.LogCaptureFixture) -> dict[str, Any]:
    [usage] = [
        r for r in caplog.records if vars(r).get("otel.event.name") == "agent.usage"
    ]
    return vars(usage)


@pytest.fixture
def run_agent(
    scripted_agent: ScriptedAgent,
    entry_point: EntryPoint,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> Callable[..., Any]:
    """Run a real agent, through each entry point, on a model that plays ``turns``."""

    async def cost(_self: Router, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * 0.001 + output_tokens * 0.01

    monkeypatch.setattr(Router, "get_conversation_cost", cost)
    caplog.set_level(logging.INFO)

    async def _run(*turns: Turn) -> tuple[Any, ScriptedModel]:
        agent, model = scripted_agent(*turns, usage=(100, 10))
        result = await entry_point(
            agent, {"websocket": None}, {"task": "Recover the robot"}
        )
        return result, model

    return _run


@pytest.mark.asyncio
async def test_usage_sums_both_invocations(
    run_agent: Callable[..., Any], caplog: pytest.LogCaptureFixture
) -> None:
    # First invocation answers in text, the second fills the report.
    result, model = await run_agent(
        "Looked at the robot.", ToolCall("ScriptedReport", {"done": True})
    )

    assert result == {"done": True}
    assert model.calls >= 2
    usage = _usage_event(caplog)
    assert usage["gen_ai.usage.input_tokens"] == 100 * model.calls
    assert usage["gen_ai.usage.output_tokens"] == 10 * model.calls
    assert usage["agent.cost"] == pytest.approx(0.2 * model.calls)


@pytest.mark.asyncio
async def test_usage_is_reported_when_the_second_invocation_fails(
    run_agent: Callable[..., Any], caplog: pytest.LogCaptureFixture
) -> None:
    with pytest.raises(RuntimeError):
        await run_agent("Looked at the robot.", RuntimeError("model unavailable"))

    # Tokens of the first invocation are still accounted for.
    assert _usage_event(caplog)["gen_ai.usage.input_tokens"] == 100


@pytest.mark.asyncio
async def test_a_run_cancelled_while_pricing_still_finishes_its_trace() -> None:
    pricing = asyncio.Event()

    async def price_forever(_input_tokens: int, _output_tokens: int) -> float:
        pricing.set()
        _ = await asyncio.Event().wait()
        return 0.0

    router = fake_router()
    router.get_conversation_cost = price_forever
    agent = make_gui_agent("unused", router=router)
    hook = FakeHook()
    run = SimpleNamespace(
        event_loop_metrics=SimpleNamespace(
            accumulated_usage={"inputTokens": 100, "outputTokens": 10}
        )
    )
    recording = asyncio.create_task(
        agent.record_usage(
            cast(StrandsAgent, cast(object, run)),
            cast(AgentLoggingHook, cast(object, hook)),
        )
    )
    _ = await pricing.wait()

    _ = recording.cancel()

    with pytest.raises(asyncio.CancelledError):
        await recording
    assert hook.trace_updates == [(True, -1.0)]
