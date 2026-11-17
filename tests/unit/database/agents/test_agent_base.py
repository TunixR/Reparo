"""The Reparo ``Agent`` as a Strands ``AgentBase``, the interface a Graph node runs."""

import json
import uuid
from collections.abc import Callable
from typing import Any

import pytest
from strands import Agent as StrandsAgent
from strands.agent import AgentBase, AgentResult

from database.agents.models import Agent
from tests.unit.fixtures.agent_fixtures import ScriptedReport
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall

type ScriptedAgent = Callable[..., tuple[Agent, ScriptedModel]]

# The first LLM phase answers in text, the structured-output phase fills the report.
TURNS = ("Looked at the robot.", ToolCall("ScriptedReport", {"done": True}))
INPUTS: dict[str, Any] = {
    "invocation_state": {"websocket": None},
    "task": "Recover the robot",
}


@pytest.mark.asyncio
async def test_stream_async_ends_with_the_structured_agent_result(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, model = scripted_agent(*TURNS, usage=(100, 10))

    events = [event async for event in agent.stream_async(**INPUTS)]

    # One result for the whole run: the two LLM phases are not separate results.
    [final] = [event for event in events if "result" in event]
    assert final is events[-1]
    result = final["result"]
    assert isinstance(result, AgentResult)
    assert result.structured_output == ScriptedReport(done=True)
    assert model.calls >= 2
    assert result.metrics.accumulated_usage["inputTokens"] == 100 * model.calls


def _graph_node(executor: AgentBase) -> AgentBase:
    """Stand-in for Strands' ``GraphBuilder.add_node``, which takes an AgentBase."""
    return executor


def test_agent_is_a_strands_agent_base(mock_agent: Agent) -> None:
    # basedpyright also checks the call: an Agent row is accepted as an AgentBase.
    assert isinstance(_graph_node(mock_agent), AgentBase)


@pytest.mark.asyncio
async def test_invoke_async_returns_the_final_agent_result(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, _ = scripted_agent(*TURNS)

    result = await agent.invoke_async(**INPUTS)

    assert isinstance(result, AgentResult)
    assert result.structured_output == ScriptedReport(done=True)


@pytest.mark.asyncio
async def test_run_returns_the_structured_response_as_a_dict(
    scripted_agent: ScriptedAgent,
) -> None:
    agent, _ = scripted_agent(*TURNS)

    assert await agent.run(**INPUTS) == {"done": True}


def test_sync_call_returns_the_agent_result(scripted_agent: ScriptedAgent) -> None:
    agent, _ = scripted_agent(*TURNS)

    result = agent(**INPUTS)

    assert result.structured_output == ScriptedReport(done=True)


def test_output_model_is_the_response_model(
    scripted_agent: ScriptedAgent, mock_agent: Agent
) -> None:
    agent, _ = scripted_agent(*TURNS)

    assert agent.output_model() is ScriptedReport
    assert mock_agent.output_model() is None


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
