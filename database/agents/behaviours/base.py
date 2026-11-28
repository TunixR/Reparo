"""What every agent type shares, and the registry from `AgentType` to its class.

An agent row says what an agent is; its behaviour class says how it runs. Each
concrete class registers itself for one `AgentType` (``agent_type=`` in its class
statement), so only this package branches on the type: everything else asks
`behaviour_of` for the class and `validate_agent` for the type's rules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel
from strands import Agent as StrandsAgent
from strands import ToolContext, tool
from strands.tools.decorator import DecoratedFunctionTool

from agent_tools.hooks import AgentLoggingHook
from database.agent_graphs.runner import (
    AGENT_GRAPH_CONTEXT_KEY,
    AgentGraphContext,
    NodeRun,
)
from database.agents.models import Agent, AgentType, IgnoredField, ModelClass
from database.provider.pricing import UNKNOWN_COST
from observability import attributes as attrs
from observability.events import log_event

logger = logging.getLogger(__name__)

_UNBOUND = object()
"""What the graph context gives for an argument it has no value for."""

_REGISTRY: dict[AgentType, type[AgentBehaviour]] = {}


class AgentBehaviour(ABC):
    """How an agent of one type runs: as a tool of its parent, standalone, or as a
    graph node. The three ways bind the inputs the same way and differ only in how
    they hand the run its inputs and what they return."""

    agent_type: ClassVar[AgentType | None] = None
    """The type this class runs; None for an abstract class."""

    IGNORED_FIELDS: ClassVar[tuple[IgnoredField, ...]] = ()
    """The fields an agent of this type stores but doesn't use."""

    def __init_subclass__(
        cls, agent_type: AgentType | None = None, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.agent_type = agent_type
        if agent_type is None:
            return
        if agent_type in _REGISTRY:
            raise TypeError(
                f"{agent_type.value} already runs as {_REGISTRY[agent_type].__name__}."
            )
        _REGISTRY[agent_type] = cls

    def __init__(self, agent: Agent) -> None:
        self.agent: Agent = agent

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @classmethod
    def validate(cls, agent: Agent) -> list[IgnoredField]:
        """Check ``agent`` against this type's rules and return the fields it ignores.

        A type may first replace fields it fixes (see `GuiAgent`). Raises
        `ValueError` for an agent the type rejects.
        """
        cls.validate_parser_path(agent.parser_path)
        router = agent.router  # pyright: ignore[reportUnnecessaryComparison] - unset until assigned
        if router is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError("The agent has no router.")
        Agent.validate_capabilities_support_input_type(
            router.capabilities, agent.input_type, router_name=router.model_name
        )
        return list(cls.IGNORED_FIELDS)

    @classmethod
    def validate_parser_path(cls, parser_path: str | None) -> None:
        """Reject a parser path: only a `TextGuiAgent` parses its replies."""
        if parser_path is not None:
            raise ValueError(
                "Only a TextGuiAgent takes a parser_path; set it to null for "
                f"{cls.type_name()}."
            )

    @classmethod
    def type_name(cls) -> str:
        return cls.agent_type.value if cls.agent_type is not None else cls.__name__

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    @property
    def tool_name(self) -> str:
        """The name the agent has as a tool of its parent, from its row."""
        return self.agent.name.lower().replace(" ", "_")

    @abstractmethod
    def output_class(self) -> ModelClass | None:
        """The class of the agent's structured output, None if it has none."""

    @abstractmethod
    async def run(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> BaseModel:
        """Run the agent on bound inputs and return its structured output."""

    def bind_inputs(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate ``inputs``, record them in ``invocation_state`` for the agent
        trace, and return them."""
        bound = self.agent.validate_input(**inputs)
        invocation_state["inputs"] = bound
        return bound

    async def __call__(
        self, invocation_state: dict[str, Any] | None = None, **inputs: Any
    ) -> dict[str, Any]:
        """Run the agent on explicit inputs and return its structured output as a dict."""
        if invocation_state is None:
            invocation_state = {}
        bound = self.bind_inputs(invocation_state, inputs)
        return (await self.run(invocation_state, bound)).model_dump()

    async def run_node(
        self, invocation_state: dict[str, Any], input_map: Mapping[str, str]
    ) -> NodeRun:
        """Run the agent as a node of an agent graph and return its structured output.

        Each argument comes from the graph context at its ``input_map`` path, else by
        its name.
        """
        context: AgentGraphContext = invocation_state[AGENT_GRAPH_CONTEXT_KEY]
        # An argument nothing in the context has stays unbound, so validation reports
        # it as missing; a None found in the context is a value.
        inputs = {
            a.name: value
            for a in self.agent.arguments
            if (value := context.get(input_map.get(a.name, a.name), _UNBOUND))
            is not _UNBOUND
        }
        bound = self.bind_inputs(invocation_state, inputs)
        return NodeRun(output=await self.run(invocation_state, bound))

    async def run_tool(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> str | dict[str, Any]:
        """What the agent's tool returns to its parent."""
        return await self(invocation_state, **inputs)

    def as_tool(self) -> DecoratedFunctionTool:  # pyright: ignore[reportMissingTypeArgument]
        """The agent as a tool of another agent: its name and description come from
        its row, its input schema from its arguments."""

        @tool(
            name=self.tool_name,
            description=self.agent.description,
            inputSchema=self.agent.get_input_schema(),
            context=True,
        )
        async def agent_tool(tool_context: ToolContext) -> str | dict[str, Any]:
            state = tool_context.invocation_state
            if "websocket" not in state:
                raise ValueError(
                    "WebSocket must be provided in tool context invocation state."
                )
            if "robot_exception_id" not in state:
                raise ValueError(
                    "A valid robot_exception_id must be provided in tool context invocation state."
                )
            return await self.run_tool(state, tool_context.tool_use.get("input", {}))

        return agent_tool

    # ------------------------------------------------------------------
    # Traces and usage
    # ------------------------------------------------------------------

    def logging_hook(self, invocation_state: dict[str, Any]) -> AgentLoggingHook:
        """The hook that writes this run's agent trace, under the caller's trace."""
        return AgentLoggingHook(
            agent_id=self.agent.id,
            invocation_state=invocation_state,
            parent_trace_id=invocation_state.get("parent_trace_id"),
        )

    async def record_usage(
        self, strands_agent: StrandsAgent, hook: AgentLoggingHook | None
    ) -> None:
        """Finish the agent trace with the run's cost and log its token usage.

        Every AgentResult.metrics is the same object, so the usage sums every
        invocation of ``strands_agent``, including tokens spent before a failure.
        Pricing may wait on the provider's catalogue; a run cancelled meanwhile
        still finishes its trace, at `UNKNOWN_COST`.
        """
        usage = strands_agent.event_loop_metrics.accumulated_usage
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        cost = UNKNOWN_COST
        try:
            cost = await self.agent.router.get_conversation_cost(
                input_tokens, output_tokens
            )
        finally:
            if hook is not None:
                hook.update_trace(finished=True, cost=cost)
        log_event(
            logger,
            attrs.EVENT_AGENT_USAGE,
            {
                attrs.AGENT_ID: self.agent.id,
                attrs.GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
                attrs.GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
                attrs.AGENT_COST: cost,
            },
        )


def behaviour_class(agent_type: AgentType | str) -> type[AgentBehaviour]:
    """The class that runs agents of ``agent_type``."""
    return _REGISTRY[AgentType(agent_type)]


def behaviour_of(agent: Agent) -> AgentBehaviour:
    """How ``agent`` runs, given its type."""
    return behaviour_class(agent.type)(agent)


def validate_agent(agent: Agent) -> list[IgnoredField]:
    """Run the rules of ``agent``'s type on it and return the fields the type ignores.

    The create, update and seed paths call it once the fields are set, before
    committing, whether the row came from the constructor or `model_validate`.
    Raises `ValueError` for an agent its type rejects.
    """
    return behaviour_class(agent.type).validate(agent)


def agent_types_of(kind: type[AgentBehaviour]) -> frozenset[AgentType]:
    """The agent types whose class is ``kind`` or a subclass of it."""
    return frozenset(t for t, cls in _REGISTRY.items() if issubclass(cls, kind))
