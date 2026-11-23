"""Context compression for recovery agents: old messages become a summary, and each one is reported.

``ReportingSummarizingManager`` is Strands' ``SummarizingConversationManager``
with a recovery-aware summary prompt and three additions Strands leaves out:

- **Reporting.** Strands inserts the summary into ``agent.messages`` by slice
  assignment and fires no hook event, so it is invisible to the trace. Each
  summary is reported to ``on_compression`` (the agent's ``AgentLoggingHook``)
  and as a span event; one that failed while the run goes on, as a WARN log event.
- **Usage.** The summary is a model call outside the agent loop. Strands adds
  its tokens to the request's ``InvocationContext`` but not to
  ``event_loop_metrics``, which Reparo prices an agent from; they are added there.
- **No blocking.** Strands summarizes from synchronous code, proactively from a
  hook and after an overflow from the agent loop, and waits for the summary call
  on a worker thread, which stalls the event loop, and every recovery and robot
  socket on it, for the length of the call. Both run from async hooks here
  instead, at the same 70% threshold and before Strands' overflow handler, and
  wait for the summary on a thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, override

from strands.agent.conversation_manager import SummarizingConversationManager
from strands.agent.conversation_manager.conversation_manager import (
    DEFAULT_COMPRESSION_THRESHOLD,
)
from strands.agent.invocation_context import InvocationContext
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookRegistry
from strands.types.content import Message
from strands.types.event_loop import Usage
from strands.types.exceptions import ContextWindowOverflowException

from observability.instrumentation.context_compression import (
    context_compressed,
    context_compression_failed,
)

if TYPE_CHECKING:
    from strands import Agent as StrandsAgent

SUMMARY_RATIO = 0.5
"""Share of the history one summary replaces. Each summary costs a model call over
the replaced range and moves the prompt cache's break point to the summary, so a
summary should bring a context at the 70% threshold well below it: replacing half
of the messages leaves room for many more steps before the next one. Strands'
default of 0.3 brings it to about 50%, and the next summary comes a few steps later."""

RECOVERY_SUMMARIZATION_PROMPT = """\
You summarize the older part of a conversation between a recovery agent and its \
tools. The agent is recovering a failed RPA robot run and works towards one task. \
Your summary replaces the messages you are given, and the agent reads it to carry \
on with the same task: whatever you leave out is lost.

Write these sections, as short bullet points:

## Task
The task and what counts as success, in one or two lines, as the messages state it.

## Steps and outcomes
Every step or tool call, in order, with its outcome: succeeded, failed (with the \
error), or unknown. Merge identical repeated steps and say how many times they ran.

## Facts found
Values, identifiers, names, field contents, variable changes and other facts the \
steps discovered, exactly as written.

## Current state
The state of the application or screen after the last of these messages: the \
window, page or dialog that is open and what it last showed.

## Remaining
What is still left to do to finish the task, and approaches that failed and \
should not be repeated.

Rules:
- Use only what the messages say; mark anything uncertain instead of guessing.
- Keep exact values (numbers, ids, text typed or read) unchanged.
- An omitted screenshot is a placeholder: don't describe what you can't see.
- Don't address anyone, don't comment on the summary, and don't call tools."""

CONTEXT_COMPRESSION_ROLE = "context_compression"
"""The role of the marker a summary leaves in ``AgentTrace.messages``. It is never a
Strands role: the marker is written to the trace only, not to ``agent.messages``."""

type CompressionTrigger = Literal["proactive", "overflow"]


@dataclass(frozen=True, slots=True)
class ContextCompression:
    """One summary that replaced the oldest unpinned messages of an agent's history."""

    trigger: CompressionTrigger
    """``proactive`` before a model call past the threshold, ``overflow`` after the
    provider rejected a call as too long."""
    summary: str
    messages_replaced: int
    """Messages the summary took the place of, an earlier summary included."""
    tokens_before: int | None
    """The model call's projected input tokens, as Strands estimated them; ``None``
    after an overflow."""
    tokens_after: int | None
    """``tokens_before`` less the model's estimate of the replaced messages, plus
    its estimate of the summary."""
    input_tokens: int
    """Tokens of the summary call."""
    output_tokens: int


class ReportingSummarizingManager(SummarizingConversationManager):
    """Summarizes the oldest unpinned messages when the context fills, with the agent's own model.

    Args:
        pin_first: Messages at the start of the history that are never summarized.
        preserve_recent_messages: Latest messages that are never summarized.
        on_compression: Called with each summary that replaced messages.
    """

    def __init__(
        self,
        *,
        pin_first: int,
        preserve_recent_messages: int = 10,
        on_compression: Callable[[ContextCompression], None] | None = None,
    ) -> None:
        # Proactive compression is this class's own async hook; Strands' is left off.
        super().__init__(
            summary_ratio=SUMMARY_RATIO,
            preserve_recent_messages=preserve_recent_messages,
            summarization_system_prompt=RECOVERY_SUMMARIZATION_PROMPT,
            pin_first=pin_first,
            proactive_compression=None,
        )
        self.on_compression = on_compression
        self._summary_error: Exception | None = None

    @property
    def summary_message(self) -> Message | None:
        """The latest summary in the history, if any."""
        return self.get_state()["summary_message"]

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        super().register_hooks(registry, **kwargs)
        registry.add_callback(BeforeModelCallEvent, self._compress_before_model_call)
        registry.add_callback(AfterModelCallEvent, self._compress_after_overflow)

    async def _compress_before_model_call(self, event: BeforeModelCallEvent) -> None:
        tokens = event.projected_input_tokens
        if (
            tokens is None
            or event.agent.model.estimate_utilization(tokens)
            < DEFAULT_COMPRESSION_THRESHOLD
        ):
            return
        await self._compress(event.agent, event.invocation_context, tokens)

    async def _compress_after_overflow(self, event: AfterModelCallEvent) -> None:
        if not isinstance(event.exception, ContextWindowOverflowException):
            return
        # Retried here, so Strands' synchronous overflow handler isn't reached.
        # A summary that can't shrink the context raises, and the run fails.
        await self._compress(
            event.agent, event.invocation_context, None, overflow=event.exception
        )
        event.retry = True

    async def _compress(
        self,
        agent: StrandsAgent,
        context: InvocationContext | None,
        tokens_before: int | None,
        overflow: Exception | None = None,
    ) -> None:
        """Summarize the oldest messages off the event loop, and report the result.

        Strands swallows a proactive summary's error, and after an overflow falls
        back to truncating the largest text blocks: the run goes on, and the summary
        is reported as failed. Any other error after an overflow ends the run.
        """
        trigger: CompressionTrigger = "proactive" if overflow is None else "overflow"
        before = list(agent.messages)
        previous_summary = self.summary_message
        used_before = _tokens_used(context)
        self._summary_error = None
        try:
            await asyncio.to_thread(
                super().reduce_context, agent, overflow, invocation_context=context
            )
        finally:
            input_tokens, output_tokens = _count_summary_usage(
                agent, context, used_before
            )
        summary = self.summary_message
        if summary is None or summary is previous_summary:
            context_compression_failed(trigger, self._summary_error)
            return
        kept = {id(message) for message in agent.messages}
        replaced = [message for message in before if id(message) not in kept]
        tokens_after = None
        if tokens_before is not None:
            tokens_after = (
                tokens_before
                - await agent.model.count_tokens(replaced)
                + await agent.model.count_tokens([summary])
            )
        compression = ContextCompression(
            trigger=trigger,
            summary="\n".join(
                block["text"] for block in summary["content"] if "text" in block
            ),
            messages_replaced=len(replaced),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        context_compressed(compression)
        if self.on_compression is not None:
            self.on_compression(compression)

    @override
    def _summarize_oldest(
        self,
        agent: StrandsAgent,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        # The one private SDK method used here: Strands swallows a proactive
        # summary's error, so it is kept for the failure event. Tests fail if a
        # Strands release renames it.
        try:
            super()._summarize_oldest(agent, invocation_context)
        except Exception as exc:
            self._summary_error = exc
            raise


def _tokens_used(context: InvocationContext | None) -> tuple[int, int]:
    if context is None:
        return 0, 0
    return context.usage["inputTokens"], context.usage["outputTokens"]


def _count_summary_usage(
    agent: StrandsAgent, context: InvocationContext | None, used_before: tuple[int, int]
) -> tuple[int, int]:
    """Add the summary call's tokens to the agent's metrics, and return them.

    Strands adds them to the request's ``InvocationContext`` only, so the
    difference since ``used_before`` is the summary call. That holds while no
    other agent shares the request's context: none does in Reparo, each agent
    and sub-agent starts a request of its own.
    """
    used_after = _tokens_used(context)
    input_tokens = used_after[0] - used_before[0]
    output_tokens = used_after[1] - used_before[1]
    if input_tokens or output_tokens:
        agent.event_loop_metrics.update_usage(
            Usage(
                inputTokens=input_tokens,
                outputTokens=output_tokens,
                totalTokens=input_tokens + output_tokens,
            )
        )
    return input_tokens, output_tokens
