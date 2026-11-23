"""`GuiAgent`: what every GUI agent does around its loop.

A GUI agent drives the robot's desktop from screenshots. It takes the fixed GUI
inputs (`GUI_ARGUMENTS`) and reports a `GuiRunReport`. Around the loop a subtype
fills in, it checks the robot, enriches the current span with ``gui.task.*``
(the Strands ``execute_tool`` span, or the graph node's), takes the first
screenshot, writes the agent trace, builds the instruction, and turns how the loop
ended into an outcome. See OBSERVABILITY.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, override

from fastapi import WebSocketDisconnect
from strands import Agent as StrandsAgent
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.content import Messages

from agent_tools.hooks import MaxGuiImagesHook
from agent_tools.image import capture_screenshot
from database.agents.behaviours.base import AgentBehaviour
from database.agents.behaviours.gui.shared import (
    DEFAULT_GUI_ACTIONS,
    GUI_ARGUMENTS,
    GuiActions,
    GuiRun,
    GuiRunReport,
)
from database.agents.models import Agent, IgnoredField, ModelClass
from observability.instrumentation.gui_agent import GuiTaskTelemetry
from robot.capabilities import require_vision


class GuiAgent(AgentBehaviour, ABC):
    """The parent of both GUI loops; a subtype fills in `_drive`."""

    actions: ClassVar[GuiActions] = DEFAULT_GUI_ACTIONS
    """The actions this agent can take."""

    IGNORED_FIELDS: ClassVar[tuple[IgnoredField, ...]] = (
        IgnoredField(
            field="arguments",
            reason="a GUI agent always takes the fixed GUI inputs, which replace the given ones",
        ),
        IgnoredField(
            field="input_type",
            reason="a GUI agent always works from screenshots, so it is IMAGETEXT and its router needs VISION",
        ),
        IgnoredField(
            field="output_model",
            reason="a GUI agent always reports a GuiRunReport",
        ),
        IgnoredField(
            field="tools",
            reason="a GUI agent's actions are built in; link it to its parent as a sub-agent instead",
        ),
        IgnoredField(
            field="sub_agents",
            reason="a GUI agent doesn't call other agents",
        ),
    )

    @classmethod
    @override
    def validate(cls, agent: Agent) -> list[IgnoredField]:
        agent.arguments = list(GUI_ARGUMENTS)
        agent.input_type = Agent.InputType.IMAGETEXT
        return super().validate(agent)

    @override
    def output_class(self) -> ModelClass:
        return GuiRunReport

    @override
    async def run(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> GuiRunReport:
        """Drive the robot's desktop on bound inputs and report what happened.

        Every error propagates, so a robot disconnect or a model failure ends a graph
        node instead of a report.
        """
        run = await self._session(invocation_state, inputs, contain_errors=False)
        return run.report()

    @override
    async def run_tool(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> str | dict[str, Any]:
        """The report, the model's messages and the variables as the run left them.

        An error that ended the run comes back as a string; a robot disconnect and a
        ``RuntimeError`` still propagate. The parent needs the variables, since an
        extraction stores what it obtained in them, and the messages, to judge the
        outcome from what the agent did and saw.
        """
        run = await self.drive(invocation_state, inputs, contain_errors=True)
        if run.error is not None:
            return run.error
        return run.report().model_dump() | {
            "messages": run.transcript(),
            "variables": run.variables,
        }

    async def drive(
        self,
        invocation_state: dict[str, Any],
        inputs: Mapping[str, Any],
        *,
        contain_errors: bool,
    ) -> GuiRun:
        """Bind ``inputs`` and run until the agent finishes, stops or hits the cap."""
        bound = self.bind_inputs(invocation_state, inputs)
        return await self._session(
            invocation_state, bound, contain_errors=contain_errors
        )

    async def _session(
        self,
        state: dict[str, Any],
        inputs: Mapping[str, Any],
        *,
        contain_errors: bool,
    ) -> GuiRun:
        """The loop skeleton around `_drive`.

        A robot disconnect and a ``RuntimeError`` always propagate. Any other error
        propagates too, unless ``contain_errors``: then the run returns it as ``error``.
        """
        if "websocket" not in state:
            raise ValueError("WebSocket must be provided in tool context")
        require_vision(state)

        websocket = state["websocket"]
        # In a recovery the variables come from code, so the robot gets back what
        # alter_variable changed; standalone, the inputs hold them.
        state_variables = state.get("variables")
        variables: dict[str, Any] = (
            state_variables  # pyright: ignore[reportUnknownVariableType]
            if isinstance(state_variables, dict)
            else inputs["variables"]
        )
        telemetry = GuiTaskTelemetry(
            task=inputs["task"],
            model=inputs["model"],
            ui_log=inputs["ui_log"],
            errored_act=inputs["errored_act"],
        )
        image, image_size = await capture_screenshot(websocket, telemetry)
        hook = self.logging_hook(state)
        run = GuiRun(
            actions=self.actions,
            telemetry=telemetry,
            websocket=websocket,
            hook=hook,
            router=self.agent.router,
            variables=variables,
            image=image,
            image_size=image_size,
        )
        agent = StrandsAgent(  # type: ignore
            model=self.agent.get_model(),
            tools=self._tools(run),
            hooks=[hook, MaxGuiImagesHook()],
        )
        run.agent = agent
        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {"text": self._prompt(self._instruction(run, inputs, variables))},
                    {"image": {"format": "jpeg", "source": {"bytes": image}}},
                ],
            }
        ]

        try:
            await self._drive(run, agent, messages, dict(state))
            if run.disconnect is not None:
                raise run.disconnect
            if telemetry.outcome is None:
                telemetry.model_stopped(telemetry.actions)
            return run
        except WebSocketDisconnect:
            telemetry.client_disconnected(telemetry.actions)
            raise
        except RuntimeError as e:
            telemetry.failed(telemetry.actions, e, escaped=True)
            raise
        except Exception as e:
            telemetry.failed(telemetry.actions, e, escaped=not contain_errors)
            if not contain_errors:
                raise
            run.error = str(e)
            return run
        finally:
            self.record_usage(agent, hook)

    def _instruction(
        self, run: GuiRun, inputs: Mapping[str, Any], variables: dict[str, Any]
    ) -> str:
        # Actions map coordinates back from this space (Router.original_coords).
        model_width, model_height = run.model_space()
        return f"""
    Task: {inputs["task"]}
    UI Log: {inputs["ui_log"]}
    Errored Activity: {inputs["errored_act"]}
    Variables: {variables}
    Model Context: {inputs["model"]}

    IMPORTANT: Give coordinates in a {model_width}x{model_height} space: x from 0 to {model_width} (left to right), y from 0 to {model_height} (top to bottom).
    """

    def _prompt(self, instruction: str) -> str:
        """The first message's text: the agent's prompt with its ``{instruction}``."""
        return self.agent.prompt.format(instruction=instruction)

    def _tools(self, run: GuiRun) -> list[DecoratedFunctionTool]:  # pyright: ignore[reportMissingTypeArgument, reportUnusedParameter]
        """The Strands tools the loop's model gets; none by default."""
        return []

    @abstractmethod
    async def _drive(
        self,
        run: GuiRun,
        agent: StrandsAgent,
        messages: Messages,
        state: dict[str, Any],
    ) -> None:
        """Run ``agent`` (the run's model) on ``messages`` and perform its actions
        until ``run.over`` or the model stops."""
