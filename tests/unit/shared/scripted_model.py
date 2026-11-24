"""Strands model that replays a fixed script, so agent loops run without an LLM."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, override

from pydantic import BaseModel
from strands.models import Model

from agent_tools.context_compression import RECOVERY_SUMMARIZATION_PROMPT


@dataclass(frozen=True)
class ToolCall:
    """A model turn that requests one tool call."""

    name: str
    input: dict[str, Any]


type Turn = str | ToolCall | Exception

type Summary = str | Exception | Callable[[], str]
"""The answer to a summary request: text, a failure, or a call made when asked."""


class ScriptedModel(Model):
    """Answers each model call with the next turn of a script.

    Text ends the agent loop, a ``ToolCall`` makes Strands run that tool (or
    fill a ``structured_output_model``, whose tool is named after the class),
    and an exception is raised as a model failure. The last turn is repeated
    if the agent asks for more. Every answer reports ``usage`` tokens.

    A summary request of the recovery conversation manager (its system prompt
    is ``RECOVERY_SUMMARIZATION_PROMPT``) doesn't take a turn: it is answered
    with ``summary`` and reports ``summary_usage`` tokens. The model's window is
    ``context_window_limit`` tokens, unknown when None.
    """

    def __init__(
        self,
        *turns: Turn,
        usage: tuple[int, int] = (12, 4),
        context_window_limit: int | None = None,
        summary: Summary = "Earlier steps, summarized.",
        summary_usage: tuple[int, int] = (50, 5),
    ) -> None:
        assert turns, "a script needs at least one turn"
        self._turns = list(turns)
        self._input_tokens, self._output_tokens = usage
        self._context_window_limit = context_window_limit
        self._summary = summary
        self._summary_usage = summary_usage
        self.calls = 0
        self.requests: list[Any] = []
        self.tool_specs: list[Any] = []
        """The tool specs of each call, in order."""
        self.summary_requests: list[Any] = []

    @override
    def update_config(self, **model_config: Any) -> None:
        return None

    @override
    def get_config(self) -> Any:
        if self._context_window_limit is None:
            return {}
        return {"context_window_limit": self._context_window_limit}

    @override
    async def structured_output(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        output_model: type[BaseModel],
        prompt: Any,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError
        yield {}

    @override
    async def stream(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        if system_prompt == RECOVERY_SUMMARIZATION_PROMPT:
            self.summary_requests.append(messages)
            if isinstance(self._summary, Exception):
                raise self._summary
            text = self._summary if isinstance(self._summary, str) else self._summary()
            for event in _answer(text, self._summary_usage):
                yield event
            return
        self.requests.append(messages)
        self.tool_specs.append(tool_specs)
        turn = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        if isinstance(turn, Exception):
            raise turn
        for event in _answer(
            turn, (self._input_tokens, self._output_tokens), f"call-{self.calls}"
        ):
            yield event


def _answer(
    turn: str | ToolCall, usage: tuple[int, int], tool_use_id: str = ""
) -> list[dict[str, Any]]:
    """The stream events of a model answer playing ``turn`` and reporting ``usage``."""
    input_tokens, output_tokens = usage
    if isinstance(turn, ToolCall):
        tool_use = {"toolUseId": tool_use_id, "name": turn.name}
        content: list[dict[str, Any]] = [
            {"contentBlockStart": {"start": {"toolUse": tool_use}}},
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": json.dumps(turn.input)}}
                }
            },
        ]
        stop_reason = "tool_use"
    else:
        content = [
            {"contentBlockStart": {"start": {}}},
            {"contentBlockDelta": {"delta": {"text": turn}}},
        ]
        stop_reason = "end_turn"
    return [
        {"messageStart": {"role": "assistant"}},
        *content,
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": stop_reason}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": input_tokens,
                    "outputTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                "metrics": {"latencyMs": 1},
            }
        },
    ]
