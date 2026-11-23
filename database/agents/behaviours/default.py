"""The default agent: a Strands agent built from the row, with its tools and sub-agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, override

from pydantic import BaseModel
from strands import Agent as StrandsAgent
from strands.hooks import HookProvider
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.content import Messages

from agent_tools.hooks import (
    AgentLoggingHook,
    LimitToolCounts,
    RequireToolCalls,
    ToolLoggingHook,
)
from agent_tools.image import screenshot_bytes
from database.agents.behaviours.base import AgentBehaviour, behaviour_of
from database.agents.models import Agent, AgentType, ModelClass
from database.agents.specs import build_output_model


class DefaultAgent(AgentBehaviour, agent_type=AgentType.Agent):
    """Runs on its inputs with its tools and sub-agents, then reports its output model.

    An agent's role in a recovery (gateway, inferer…) comes from its place in the
    graph, not from its type.
    """

    @override
    def output_class(self) -> ModelClass | None:
        if self.agent.output_model is None:
            return None
        return build_output_model(self.agent.output_model)

    def tool_limiter(self) -> LimitToolCounts:
        """A hook that enforces the call limits of the agent's tool and sub-agent links."""
        tool_limits: dict[str, int] = {}
        for at in self.agent.tools:
            if at.limit is not None:
                tool_limits[at.tool.name] = at.limit
        for sa in self.agent.sub_agents:
            if sa.limit is not None:
                tool_limits[behaviour_of(sa.child_agent).tool_name] = sa.limit
        return LimitToolCounts(tool_limits)

    def required_tools_hook(self) -> RequireToolCalls:
        """A hook that sends the agent back until its required tools are called."""
        return RequireToolCalls(
            [at.tool.name for at in self.agent.tools if at.required]
        )

    def logging_hooks(self, invocation_state: dict[str, Any]) -> list[HookProvider]:
        """The agent trace, which its sub-agents nest under, and its tool traces."""
        agent_logging_hook = self.logging_hook(invocation_state)
        invocation_state["parent_trace_id"] = agent_logging_hook.agent_trace_id
        tools = {at.tool.name: at.tool.id for at in self.agent.tools}
        tool_logging_hook = ToolLoggingHook(
            agent_trace_id=agent_logging_hook.agent_trace_id, tools=tools
        )
        return [agent_logging_hook, tool_logging_hook]

    @override
    async def run(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> BaseModel:
        """Run the agent on bound inputs and return its structured output.

        Tools, sub-agents, hooks and the output model are computed per call. The agent
        runs twice on one conversation: a free run on the inputs, then a
        structured-output call. Usage and cost of both are logged, even on failure.
        """
        agent: Agent = self.agent
        tools: list[DecoratedFunctionTool] = [  # pyright: ignore[reportMissingTypeArgument]
            t.get_tool_function() for t in agent.get_tools()
        ]
        sub_agent_tools: list[DecoratedFunctionTool] = [  # pyright: ignore[reportMissingTypeArgument]
            behaviour_of(sa).as_tool() for sa in agent.get_sub_agents()
        ]

        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {
                                "bytes": await screenshot_bytes(
                                    invocation_state["websocket"]
                                ),
                            },
                        },
                    },
                ]
                if agent.input_type == Agent.InputType.IMAGETEXT
                else [
                    {
                        "text": agent.prompt,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "I understand the instructions. I will proceed once you give me all neccesary values.",
                    }
                ],
            },
        ]

        hooks = [
            self.tool_limiter(),
            self.required_tools_hook(),
            *self.logging_hooks(invocation_state),
        ]

        strands_agent = StrandsAgent(
            model=agent.get_model(),
            tools=tools + sub_agent_tools,
            hooks=hooks,
            system_prompt=agent.prompt,
            messages=messages,  # type: ignore This works fine
        )

        try:
            instruction = "\n".join(
                [f"{arg.name}: {inputs.get(arg.name)}" for arg in agent.arguments]
            )

            _ = await strands_agent.invoke_async(
                instruction,
                invocation_state=invocation_state,
            )

            response = (
                await strands_agent.invoke_async(
                    "Given our conversation so far, please provide a report using structured output with the given tool.",
                    structured_output_model=self.output_class(),
                    invocation_state=invocation_state,
                )
            ).structured_output

            if response is None:
                raise ValueError("Agent did not return a structured response.")
            return response
        finally:
            hook = next(
                (h for h in hooks if isinstance(h, AgentLoggingHook)),
                None,
            )
            self.record_usage(strands_agent, hook)
