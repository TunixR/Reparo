"""The Reparo ``Agent``'s entry points: standalone ``__call__`` and the graph's ``run_node``."""

import json
import uuid
from typing import Any

import pytest
from pydantic import BaseModel
from sqlmodel import Session
from strands import Agent as StrandsAgent
from strands import tool

from database.agent_graphs.runner import AgentGraphContext, NodeRun
from database.agents.models import Agent, AgentTool
from database.tools.models import Tool
from tests.unit.fixtures.agent_fixtures import (
    EntryPoint,
    ScriptedAgent,
    ScriptedReport,
    node_state,
)
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall

# The first LLM phase answers in text, the structured-output phase fills the report.
TURNS = ("Looked at the robot.", ToolCall("ScriptedReport", {"done": True}))
INPUTS: dict[str, Any] = {"task": "Recover the robot"}


class _Task(BaseModel):
    task: str


@tool
def ping() -> str:
    """Answer pong."""
    return "pong"


def _with_ping(session: Session, agent: Agent, **link: Any) -> Agent:
    """Give the agent the ``ping`` tool with the given limit and required flag."""
    ping_tool = Tool(
        name="ping",
        description="Answer pong.",
        fn_module=f"{__name__}.ping",
    )
    session.add(ping_tool)
    session.commit()
    session.add(AgentTool(agent_id=agent.id, tool_id=ping_tool.id, **link))
    session.commit()
    session.refresh(agent)
    return agent


def _tool_results(model: ScriptedModel) -> list[str]:
    """Texts of the tool results the agent sent back to its model."""
    return [
        content["text"]
        for message in model.requests[-1]
        for block in message["content"]
        if "toolResult" in block
        for content in block["toolResult"]["content"]
        if "text" in content
    ]


def _user_texts(model: ScriptedModel) -> list[str]:
    return [
        block["text"]
        for message in model.requests[-1]
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    ]


@pytest.mark.asyncio
async def test_run_node_returns_what_call_returns_as_a_model(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, _ = scripted_agent(*TURNS)
    as_dict = await agent(invocation_state={"websocket": None}, **INPUTS)

    agent, _ = scripted_agent(*TURNS)
    run = await agent.run_node(node_state(INPUTS, websocket=None), {})

    assert isinstance(run, NodeRun)
    assert isinstance(run.output, ScriptedReport)
    assert run.output == ScriptedReport(done=True)
    assert run.output.model_dump() == as_dict == {"done": True}


def test_output_model_is_the_response_model(
    scripted_agent: ScriptedAgent, mock_agent: Agent
) -> None:
    agent, _ = scripted_agent(*TURNS)

    assert agent.output_model() is ScriptedReport
    assert mock_agent.output_model() is None


@pytest.mark.asyncio
async def test_run_node_validates_its_inputs(scripted_agent: ScriptedAgent) -> None:
    agent, model = scripted_agent(*TURNS)

    with pytest.raises(ValueError, match="Missing required argument: task"):
        _ = await agent.run_node(node_state({}, websocket=None), {})
    assert model.calls == 0


@pytest.mark.asyncio
async def test_tool_limits_apply(
    scripted_agent: ScriptedAgent, entry_point: EntryPoint, session: Session
) -> None:
    agent, model = scripted_agent(
        ToolCall("ping", {}),
        ToolCall("ping", {}),
        "Done.",
        ToolCall("ScriptedReport", {"done": True}),
    )
    agent = _with_ping(session, agent, limit=1)

    assert await entry_point(agent, {"websocket": None}, INPUTS) == {"done": True}

    first, second = _tool_results(model)
    assert first == "pong"
    assert "throttled" in second


@pytest.mark.asyncio
async def test_required_tools_are_reminded(
    scripted_agent: ScriptedAgent, entry_point: EntryPoint, session: Session
) -> None:
    agent, model = scripted_agent(
        "Done without pinging.",
        ToolCall("ping", {}),
        "Pinged.",
        ToolCall("ScriptedReport", {"done": True}),
    )
    agent = _with_ping(session, agent, required=True)

    assert await entry_point(agent, {"websocket": None}, INPUTS) == {"done": True}

    assert any("required tool(s) `ping`" in text for text in _user_texts(model))


@pytest.mark.asyncio
async def test_traces_hang_off_the_callers_state(
    scripted_agent: ScriptedAgent,
    entry_point: EntryPoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, _ = scripted_agent(*TURNS)
    seen: list[dict[str, Any]] = []
    agent_trace_id = uuid.uuid4()

    def _logging_hooks(_self: Agent, invocation_state: dict[str, Any]) -> list[Any]:
        seen.append(dict(invocation_state))
        invocation_state["parent_trace_id"] = agent_trace_id
        return []

    monkeypatch.setattr(Agent, "get_logging_hooks", _logging_hooks)
    parent_trace_id = uuid.uuid4()
    state: dict[str, Any] = {"websocket": None, "parent_trace_id": parent_trace_id}

    _ = await entry_point(agent, state, INPUTS)

    # The agent trace is opened under the caller's trace, with the bound inputs,
    # and the agent's own tools and sub-agents trace under it.
    [opened] = seen
    assert opened["parent_trace_id"] == parent_trace_id
    assert opened["inputs"] == INPUTS
    assert state["parent_trace_id"] == agent_trace_id


@pytest.mark.asyncio
async def test_agent_as_a_tool_answers_with_its_structured_response(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, _ = scripted_agent(*TURNS)
    parent = StrandsAgent(
        model=ScriptedModel(
            ToolCall(agent.get_tool_name(), {"task": "Recover the robot"}), "Done."
        ),
        tools=[agent.as_tool()],
        callback_handler=None,
    )

    _ = await parent.invoke_async(
        "Delegate the recovery",
        invocation_state={"websocket": None, "robot_exception_id": uuid.uuid4()},
    )

    [tool_result] = [
        block["toolResult"]
        for message in parent.messages
        for block in message["content"]
        if "toolResult" in block
    ]
    assert tool_result["status"] == "success"
    assert tool_result["content"] == [{"text": json.dumps({"done": True})}]


def test_a_bare_name_is_a_recovery_input_else_the_latest_output() -> None:
    context = AgentGraphContext(recovery_inputs={"task": "from recovery"})
    context.record("first", _Task(task="older"))
    context.record("second", _Task(task="newer"))

    assert context.get("task") == "from recovery"
    assert context.get("recovery.task") == "from recovery"
    assert context.get("first.task") == "older"
    del context.recovery_inputs["task"]  # pyright: ignore[reportIndexIssue] - a dict here
    assert context.get("task") == "newer"
    assert context.get("missing") is None
    assert context.get("triage.task") is None


@pytest.mark.asyncio
async def test_run_node_binds_the_input_map_before_names(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, _ = scripted_agent(*TURNS)
    state = node_state({"task": "by name", "goal": "mapped"}, websocket=None)

    _ = await agent.run_node(state, {"task": "recovery.goal"})

    assert state["inputs"] == {"task": "mapped"}


@pytest.mark.asyncio
async def test_run_node_rejects_a_mapped_path_without_a_value(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, model = scripted_agent(*TURNS)

    with pytest.raises(ValueError, match="Missing required argument: task"):
        _ = await agent.run_node(
            node_state({}, websocket=None), {"task": "triage.task"}
        )
    assert model.calls == 0
