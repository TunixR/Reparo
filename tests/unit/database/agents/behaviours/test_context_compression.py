"""Every agent type compresses its context through one shared conversation manager."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, override
from uuid import uuid4

import pytest
from sqlmodel import Session
from strands import tool
from strands.hooks import HookRegistry

from agent_tools.context_compression import RecoveryConversationManager
from agent_tools.hooks import AgentLoggingHook
from database.agents.behaviours import (
    DefaultAgent,
    GuiAgent,
    TextGuiAgent,
    behaviour_of,
)
from database.agents.models import Agent, AgentTool, AgentType
from database.provider.models import Router
from database.tools.models import Tool
from tests.unit.fixtures.agent_fixtures import EntryPoint, ScriptedAgent
from tests.unit.shared.gui_fakes import (
    GUI_KINDS,
    UITARS_PARSER,
    FakeHook,
    drive,
    make_gui_agent,
    make_robot,
    patch_gui_io,
    scripted_model,
)
from tests.unit.shared.log_events import logged_events
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn


@pytest.mark.parametrize(
    ("agent_type", "pinned", "recent"),
    [
        pytest.param(AgentType.Agent, 3, 10, id="Agent"),
        pytest.param(AgentType.TooledGuiAgent, 1, 12, id="TooledGuiAgent"),
        pytest.param(AgentType.TextGuiAgent, 1, 12, id="TextGuiAgent"),
    ],
)
def test_each_type_pins_its_task_and_reports_to_its_trace(
    make_router: Callable[..., Router],
    agent_type: AgentType,
    pinned: int,
    recent: int,
) -> None:
    agent = Agent(
        name="An Agent",
        description="d",
        prompt="{instruction}",
        type=agent_type,
        parser_path=UITARS_PARSER if agent_type is AgentType.TextGuiAgent else None,
        router=make_router(persist=False),
    )
    hook = AgentLoggingHook(agent_id=uuid4(), invocation_state={})

    manager = behaviour_of(agent).conversation_manager(hook)

    assert isinstance(manager, RecoveryConversationManager)
    assert manager.pin_first == pinned
    assert manager.preserve_recent_messages == recent
    assert manager.on_compression == hook.log_context_compression


# ---------------------------------------------------------------------------
# A generic agent
# ---------------------------------------------------------------------------

PINGS = [ToolCall("ping", {})] * 8
"""Enough steps to fill more than the 10 latest messages a summary keeps."""
REPORT = ToolCall("ScriptedReport", {"done": True})
WINDOW = 1_000
FULL = (800, 4)
"""Usage past 70% of `WINDOW`: every later model call is due for a summary."""


@tool
def ping() -> str:
    """Answer pong, at length."""
    return "pong " * 100


class _UnsavedHook(AgentLoggingHook):
    """An ``AgentLoggingHook`` that keeps its trace messages instead of writing rows."""

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        return None

    @override
    def update_trace(self, finished: bool = False, cost: float = 0.0) -> None:
        return None


@pytest.fixture
def pinging_agent(
    scripted_agent: ScriptedAgent,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Agent, ScriptedModel, list[_UnsavedHook]]:
    """An agent with the ``ping`` tool whose every answer reports a full context."""
    agent, model = scripted_agent(
        *PINGS, "Pinged.", REPORT, usage=FULL, context_window_limit=WINDOW
    )
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
    hooks: list[_UnsavedHook] = []

    def logging_hooks(self: DefaultAgent, state: dict[str, Any]) -> list[_UnsavedHook]:
        hooks.append(_UnsavedHook(agent_id=self.agent.id, invocation_state=state))
        return hooks[-1:]

    monkeypatch.setattr(DefaultAgent, "logging_hooks", logging_hooks)
    return agent, model, hooks


@pytest.mark.asyncio
async def test_a_generic_agent_keeps_its_prompt_and_instruction_through_a_summary(
    pinging_agent: tuple[Agent, ScriptedModel, list[_UnsavedHook]],
    entry_point: EntryPoint,
) -> None:
    agent, model, hooks = pinging_agent

    result = await entry_point(agent, {"websocket": None}, {"task": "Recover it"})

    assert result == {"done": True}
    first_three = model.requests[0][:3]
    assert first_three[0]["content"] == [{"text": agent.prompt}]
    assert first_three[2]["content"] == [{"text": "task: Recover it"}]
    assert all(request[:3] == first_three for request in model.requests)
    assert model.requests[-1][3]["content"] == [{"text": "Earlier steps, summarized."}]
    [hook] = hooks
    markers = [m for m in hook.messages if m["role"] == "context_compression"]
    assert len(markers) == len(model.summary_requests) > 0


@pytest.mark.asyncio
async def test_the_summary_tokens_are_priced_with_the_run(
    pinging_agent: tuple[Agent, ScriptedModel, list[_UnsavedHook]],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(
        Router, "get_conversation_cost", lambda _self, i, o: i * 0.001 + o * 0.01
    )
    agent, model, _ = pinging_agent

    _ = await behaviour_of(agent)(invocation_state={"websocket": None}, task="Go")

    [usage] = logged_events(caplog, "agent.usage")
    summaries = len(model.summary_requests)
    assert summaries > 0
    input_tokens = FULL[0] * model.calls + 50 * summaries
    output_tokens = FULL[1] * model.calls + 5 * summaries
    assert vars(usage)["gen_ai.usage.input_tokens"] == input_tokens
    assert vars(usage)["agent.cost"] == pytest.approx(
        input_tokens * 0.001 + output_tokens * 0.01
    )


# ---------------------------------------------------------------------------
# Both GUI loops
# ---------------------------------------------------------------------------

STEPS = 9
"""Clicks before finishing: more steps than the latest ones a summary keeps."""


def _gui_turns(kind: type[GuiAgent]) -> list[Turn]:
    if kind is TextGuiAgent:
        click: Turn = "Action: click(point='<point>1 1</point>')"
        return [*[click] * STEPS, "Action: finished(content='done')"]
    click = ToolCall("click", {"start_box": [1, 1]})
    return [*[click] * STEPS, ToolCall("finished", {"content": "done"})]


@pytest.mark.parametrize("kind", GUI_KINDS)
@pytest.mark.asyncio
async def test_a_gui_loop_keeps_its_task_message_through_a_summary(
    monkeypatch: pytest.MonkeyPatch, kind: type[GuiAgent]
) -> None:
    patch_gui_io(monkeypatch)
    gui_agent = make_gui_agent(
        *_gui_turns(kind),
        kind=kind,
        prompt="{actions}\n{instruction}" if kind is TextGuiAgent else "{instruction}",
        parser_path=UITARS_PARSER if kind is TextGuiAgent else None,
        usage=FULL,
        context_window_limit=WINDOW,
    )
    model = scripted_model(gui_agent)
    robot = make_robot(screenshots=STEPS + 1)

    run = await drive(gui_agent, robot)

    assert run.summary == "done"
    assert len(robot.actions) == STEPS
    task = model.requests[0][0]["content"][0]
    assert "Task: Fill the invoice" in task["text"]
    assert all(request[0]["content"][0] == task for request in model.requests)
    assert model.requests[-1][1]["content"] == [{"text": "Earlier steps, summarized."}]
    assert isinstance(run.hook, FakeHook)
    assert len(run.hook.compressions) == len(model.summary_requests) > 0
    assert [{"text": "Earlier steps, summarized."}] in run.transcript()
    # Each call reads only its latest state block; the summarizer reads none.
    assert all(
        str(request).count("<recovery_state>") == 1
        and "<recovery_state>" in str(request[-1])
        for request in model.requests
    )
    assert "<recovery_state>" not in str(model.summary_requests)
