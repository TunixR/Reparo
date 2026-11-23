"""`TextGuiAgent`: a GUI agent whose model writes its actions as free text.

The agent's parser (``parser_path``) describes the GUI actions in its model's
syntax, through the prompt's ``{actions}`` placeholder, and reads them back out of
each reply. The actions it reads run like any GUI action; the next message shows
the model the new screenshot, or why an action failed. A reply the parser can't
read is answered with the error and counts as a ``gui.action_retry``.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, override

from strands import Agent as StrandsAgent
from strands.agent import AgentResult
from strands.types.content import ContentBlock, Messages

from database.agents.behaviours.gui.base import GuiAgent
from database.agents.behaviours.gui.parsers import (
    ActionParseError,
    ActionParser,
    load_parser,
)
from database.agents.behaviours.gui.shared import GuiRun
from database.agents.models import AgentType

MAX_PARSE_RETRIES = 3
"""How many replies in a row the parser may fail to read before the run stops."""


class TextGuiAgent(GuiAgent, agent_type=AgentType.TextGuiAgent):
    @classmethod
    @override
    def validate_parser_path(cls, parser_path: str | None) -> None:
        if parser_path is None:
            raise ValueError("A TextGuiAgent needs a parser_path.")
        _ = load_parser(parser_path)

    @cached_property
    def parser(self) -> ActionParser:
        if self.agent.parser_path is None:
            raise ValueError(
                f"The TextGuiAgent '{self.agent.name}' has no parser_path."
            )
        return load_parser(self.agent.parser_path)

    @override
    def _prompt(self, run: GuiRun, instruction: str) -> str:
        return self.agent.prompt.format(
            instruction=instruction, actions=self.parser.describe(self.actions)
        )

    @override
    async def _drive(
        self,
        run: GuiRun,
        agent: StrandsAgent,
        messages: Messages,
        state: dict[str, Any],
    ) -> None:
        reply = await agent.invoke_async(messages, invocation_state=state)
        failed_reads = 0
        while True:
            try:
                actions = self.parser.parse(_text_of(reply))
            except ActionParseError as exc:
                failed_reads += 1
                run.telemetry.retry(run.telemetry.actions + 1, exc)
                if failed_reads > MAX_PARSE_RETRIES:
                    run.telemetry.stopped(run.telemetry.actions)
                    return
                reply = await agent.invoke_async(
                    f"Your reply could not be read: {exc} Answer in the output format, "
                    "with an action.",
                    invocation_state=state,
                )
                continue
            failed_reads = 0

            content: list[ContentBlock] = []
            for action in actions:
                outcome = await run.perform(action.name, action.inputs)
                content.extend(outcome.content)
                if outcome.error is not None or run.over:
                    break
            if run.over:
                return
            reply = await agent.invoke_async(content, invocation_state=state)


def _text_of(reply: AgentResult) -> str:
    return "".join(
        block["text"] for block in reply.message.get("content", []) if "text" in block
    )
