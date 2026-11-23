"""A generic agent summarizes a full context but keeps its prompt and instruction."""

from __future__ import annotations

import logging
from typing import Any, override

import pytest
from sqlmodel import Session
from strands import tool
from strands.hooks import HookRegistry

from agent_tools.context_compression import ContextCompression
from agent_tools.hooks import AgentLoggingHook
from database.agents.models import Agent, AgentTool
from database.provider.models import Router
from database.tools.models import Tool
from tests.unit.fixtures.agent_fixtures import ScriptedAgent
from tests.unit.shared.log_events import logged_events
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall

PINGS = [ToolCall("ping", {})] * 8
"""Enough steps to fill more than the 10 latest messages a summary keeps."""
REPORT = ToolCall("ScriptedReport", {"done": True})


@tool
def ping() -> str:
    """Answer pong, at length."""
    return "pong " * 100


class _RecordingHook(AgentLoggingHook):
    """An ``AgentLoggingHook`` that keeps the compressions instead of writing rows."""

    compressions: list[ContextCompression] = []

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        return None

    @override
    def log_context_compression(self, compression: ContextCompression) -> None:
        self.compressions.append(compression)

    @override
    def update_trace(self, finished: bool = False, cost: float = 0.0) -> None:
        return None


@pytest.fixture
def pinging_agent(
    scripted_agent: ScriptedAgent,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Agent, ScriptedModel, list[ContextCompression]]:
    """An agent with the ``ping`` tool whose every answer reports a full context."""
    agent, model = scripted_agent(*PINGS, "Pinged.", REPORT, usage=(10**9, 4))
    ping_tool = Tool(
        name="ping", description="Answer pong.", fn_module=f"{__name__}.ping"
    )
    session.add(ping_tool)
    session.commit()
    session.add(
        AgentTool(agent_id=agent.id, tool_id=ping_tool.id, limit=None, required=False)
    )
    session.commit()
    session.refresh(agent)
    compressions: list[ContextCompression] = []
    monkeypatch.setattr(_RecordingHook, "compressions", compressions)
    monkeypatch.setattr(
        Agent,
        "get_logging_hooks",
        lambda self, state: [_RecordingHook(agent_id=self.id, invocation_state=state)],
    )
    monkeypatch.setattr(
        Router, "get_conversation_cost", lambda _self, i, o: i * 0.001 + o * 0.01
    )
    return agent, model, compressions


@pytest.mark.asyncio
async def test_a_full_context_is_summarized_after_the_pinned_instruction(
    pinging_agent: tuple[Agent, ScriptedModel, list[ContextCompression]],
) -> None:
    agent, model, compressions = pinging_agent

    result = await agent(invocation_state={"websocket": None}, task="Recover it")

    assert result == {"done": True}
    # The seeded prompt and answer, then the instruction, open every request.
    first_three = model.requests[0][:3]
    assert first_three[2]["content"] == [{"text": "task: Recover it"}]
    assert all(request[:3] == first_three for request in model.requests)
    assert model.requests[-1][3]["content"] == [{"text": "Earlier steps, summarized."}]
    assert compressions
    assert compressions[0].summary == "Earlier steps, summarized."


@pytest.mark.asyncio
async def test_the_summary_tokens_are_priced_with_the_run(
    pinging_agent: tuple[Agent, ScriptedModel, list[ContextCompression]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    agent, model, _ = pinging_agent

    _ = await agent(invocation_state={"websocket": None}, task="Recover it")

    [usage] = logged_events(caplog, "agent.usage")
    summaries = len(model.summary_requests)
    assert summaries > 0
    input_tokens = 10**9 * model.calls + 50 * summaries
    output_tokens = 4 * model.calls + 5 * summaries
    assert vars(usage)["gen_ai.usage.input_tokens"] == input_tokens
    assert vars(usage)["agent.cost"] == pytest.approx(
        input_tokens * 0.001 + output_tokens * 0.01
    )
