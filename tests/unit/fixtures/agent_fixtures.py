"""
Agent fixtures for unit tests.
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel
from sqlmodel import Session

import database.agents.models as agent_models
from database.agent_graphs.runner import AGENT_GRAPH_CONTEXT_KEY, AgentGraphContext
from database.agents.models import Agent, AgentType, Argument
from database.provider.models import Router
from tests.unit.fixtures.agent_graph_fixtures import MakeAgentGraph, Node
from tests.unit.shared.scripted_model import ScriptedModel, Turn


@pytest.fixture
def make_agent(session: Session) -> Callable[..., Agent]:
    """Return a factory that persists an agent wired to the given router."""

    def _make_agent(
        *,
        router: Router,
        input_type: Agent.InputType = Agent.InputType.TEXT,
        name: str = "Test Agent",
        description: str = "A mock agent for testing.",
        prompt: str = "This is a test agent.",
        type: AgentType = AgentType.Agent,
        **overrides: Any,
    ) -> Agent:
        agent = Agent(
            id=uuid.uuid4(),
            name=name,
            prompt=prompt,
            input_type=input_type,
            description=description,
            router=router,
            router_id=router.id,
            enabled=True,
            type=type,
            **overrides,
        )
        session.add(agent)
        session.commit()
        session.refresh(agent)
        return agent

    return _make_agent


@pytest.fixture
def mock_agent(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    """Create a mock Agent record for testing."""
    return make_agent(router=mock_router)


_RECOVERY_ARGUMENTS = (
    ("task_name", "str", "string"),
    ("platform", "str", "string"),
    ("os", "str", "string"),
    ("variables", "dict", "object"),
    ("ui_log", "list", "array"),
    ("errored_act", "dict", "object"),
    ("model", "str", "string"),
)


def recovery_arguments() -> list[Argument]:
    """The seeded gateway's arguments: the seven recovery inputs."""
    return [
        Argument(name=name, description=name, type=type_, json_type=json_type)
        for name, type_, json_type in _RECOVERY_ARGUMENTS
    ]


@pytest.fixture
def gateway_agent(
    make_agent: Callable[..., Agent],
    mock_router: Router,
    make_agent_graph: MakeAgentGraph,
) -> Agent:
    """Create a Gateway Orchestrator agent, the one node of the default recovery graph."""
    gateway = make_agent(
        router=mock_router,
        name="Gateway Orchestrator",
        description="Routes recovery",
        prompt="prompt",
        response_model=None,
        type=AgentType.Agent,
        arguments=recovery_arguments(),
    )
    _ = make_agent_graph(Node("gateway", gateway))
    return gateway


type ScriptedAgent = Callable[..., tuple[Agent, ScriptedModel]]
"""Factory returned by ``scripted_agent``: turns in, the agent and its model out."""


class ScriptedReport(BaseModel):
    """Response model of ``scripted_agent``; a ``ToolCall("ScriptedReport", ...)`` fills it."""

    done: bool


@pytest.fixture
def scripted_agent(
    make_agent: Callable[..., Agent],
    mock_router: Router,
    monkeypatch: pytest.MonkeyPatch,
) -> ScriptedAgent:
    """Return a factory for a real agent whose model replays the given turns.

    The agent takes one ``task`` argument and answers with a ``ScriptedReport``.
    Its logging hooks and screenshots are disabled, so it runs without a robot.
    """

    async def _no_screenshot(_websocket: object) -> bytes:
        return b""

    monkeypatch.setattr(agent_models, "screenshot_bytes", _no_screenshot)
    # The logging hooks write audit rows outside the test session.
    monkeypatch.setattr(Agent, "get_logging_hooks", lambda _self, _state: [])
    # Strands rejects a blank prompt, and the instruction lists the arguments.
    task = Argument(name="task", description="Task", type="str", json_type="string")
    agent = make_agent(
        router=mock_router,
        arguments=[task],
        response_model=f"{__name__}.ScriptedReport",
    )

    def _scripted(
        *turns: Turn, usage: tuple[int, int] = (12, 4)
    ) -> tuple[Agent, ScriptedModel]:
        model = ScriptedModel(*turns, usage=usage)
        monkeypatch.setattr(Router, "get_model", lambda _self, **_params: model)
        return agent, model

    return _scripted


type EntryPoint = Callable[
    [Agent, dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]
]
"""Runs an agent on a state and inputs and returns its structured output as a dict."""


def node_state(inputs: dict[str, Any], **state: Any) -> dict[str, Any]:
    """An invocation state whose graph context holds ``inputs`` as recovery inputs,
    so ``run_node`` binds them by name."""
    return with_graph_context(state, inputs)


def with_graph_context(state: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Put a graph context holding ``inputs`` into ``state`` itself, and return it."""
    state[AGENT_GRAPH_CONTEXT_KEY] = AgentGraphContext(recovery_inputs=inputs)
    return state


async def _via_call(
    agent: Agent, state: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    return await agent(invocation_state=state, **inputs)


async def _via_run_node(
    agent: Agent, state: dict[str, Any], inputs: dict[str, Any]
) -> dict[str, Any]:
    return (
        await agent.run_node(with_graph_context(state, inputs), {})
    ).output.model_dump()


@pytest.fixture(params=[_via_call, _via_run_node], ids=["__call__", "run_node"])
def entry_point(request: pytest.FixtureRequest) -> EntryPoint:
    """Each way to run an agent: standalone ``__call__`` and the graph's ``run_node``."""
    return request.param
