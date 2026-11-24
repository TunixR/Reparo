"""`TooledGuiAgent`: a GUI agent whose model calls each action as a Strands tool.

Tool calling is standardised, so no parsing is needed: each action of
`DEFAULT_GUI_ACTIONS` becomes a tool, and each robot action's tool returns the next
screenshot. Each tool call gets the ``gui.action`` and ``gui.screenshot`` spans as
children of its Strands ``execute_tool`` span.
"""

from __future__ import annotations

from typing import Any, override

from fastapi import WebSocketDisconnect
from strands import Agent as StrandsAgent
from strands import ToolContext, tool
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.content import Messages
from strands.types.tools import ToolResult

from database.agents.behaviours.gui.base import GuiAgent
from database.agents.behaviours.gui.shared import GuiActionSpec, GuiRun
from database.agents.models import AgentType


class TooledGuiAgent(GuiAgent, agent_type=AgentType.TooledGuiAgent):
    @override
    def _tools(self, run: GuiRun) -> list[DecoratedFunctionTool]:  # pyright: ignore[reportMissingTypeArgument]
        return [_action_tool(spec, run) for spec in self.actions]

    @override
    async def _drive(
        self,
        run: GuiRun,
        agent: StrandsAgent,
        messages: Messages,
        state: dict[str, Any],
    ) -> None:
        _ = await agent.invoke_async(messages, invocation_state=state)


def _action_tool(spec: GuiActionSpec, run: GuiRun) -> DecoratedFunctionTool:  # pyright: ignore[reportMissingTypeArgument]
    """``spec`` as a tool of ``run``'s model.

    An error outcome is raised, so Strands hands it to the model as a tool error.
    Finishing, the action cap and a robot disconnect stop the model; the cap lets
    the tool calls already requested end first.
    """

    @tool(
        name=spec.name,
        description=spec.description,
        inputSchema={"json": spec.input_schema()},
        context=True,
    )
    async def action_tool(tool_context: ToolContext) -> ToolResult:
        try:
            outcome = await run.perform(
                spec.name, tool_context.tool_use.get("input", {})
            )
        except WebSocketDisconnect:
            # Strands hands tool errors to the model, so stop it; the skeleton raises
            # the disconnect again.
            tool_context.agent.cancel()
            raise
        if run.cap_reached:
            tool_context.agent.cancel(after_current_tools=True)
        elif run.summary is not None:
            tool_context.agent.cancel()
        if outcome.error is not None:
            raise RuntimeError(outcome.error)
        return {"status": "success", "content": outcome.content}  # pyright: ignore[reportReturnType] - Strands fills the toolUseId

    return action_tool
