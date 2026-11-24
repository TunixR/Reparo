"""Context compression for recovery agents: old messages become a summary, and each one is reported.

``RecoveryConversationManager`` is Strands' ``SummarizingConversationManager``
with a recovery-aware summary prompt, placeholders for the screenshots it
summarizes, and what Strands leaves out: each summary is reported (Strands fires
no hook event for it), its tokens are added to the agent's metrics, and it runs
from async hooks, since Strands' own proactive and overflow paths block the
event loop for the whole summary call.
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
from strands.agent.invocation_context import (
    InvocationContext,
    _create_auxiliary_invocation_context,  # pyright: ignore[reportPrivateUsage]
)
from strands.hooks import AfterModelCallEvent, BeforeModelCallEvent, HookRegistry
from strands.types.content import ContentBlock, Message
from strands.types.event_loop import Usage
from strands.types.exceptions import ContextWindowOverflowException

from observability.instrumentation.context_compression import (
    context_compressed,
    context_compression_failed,
)

if TYPE_CHECKING:
    from strands import Agent as StrandsAgent

SUMMARY_RATIO = 0.5
"""Share of the history one summary replaces."""

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

OMITTED_IMAGE_TEXT = "screenshot omitted to save context"
"""What a screenshot taken out of a message is replaced with."""

CONTEXT_COMPRESSION_ROLE = "context_compression"
"""The role of the marker a summary leaves in ``AgentTrace.messages``, never in
``agent.messages``."""

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


class RecoveryConversationManager(SummarizingConversationManager):
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
        preserve_recent_messages: int,
        on_compression: Callable[[ContextCompression], None] | None = None,
    ) -> None:
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

        After an overflow, a summary that can't shrink the context falls back to
        truncating the largest text blocks; any other error ends the run.
        """
        trigger: CompressionTrigger = "proactive" if overflow is None else "overflow"
        before = list(agent.messages)
        previous_summary = self.summary_message
        summary_context = _create_auxiliary_invocation_context(context)
        self._summary_error = None
        try:
            await asyncio.to_thread(
                super().reduce_context,
                agent,
                overflow,
                invocation_context=summary_context,
            )
        finally:
            input_tokens, output_tokens = _count_summary_usage(agent, summary_context)
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
        # Strands swallows a proactive summary's error; it is kept for the WARN.
        try:
            super()._summarize_oldest(agent, invocation_context)
        except Exception as exc:
            self._summary_error = exc
            raise

    @override
    def _generate_summary(
        self,
        messages: list[Message],
        agent: StrandsAgent,
        invocation_context: InvocationContext | None = None,
    ) -> Message:
        return super()._generate_summary(
            [_without_images(message) for message in messages],
            agent,
            invocation_context,
        )


def _without_images(message: Message) -> Message:
    """A copy of ``message`` with a placeholder for each screenshot, tool results' too."""
    content: list[ContentBlock] = []
    for block in message["content"]:
        if "image" in block:
            block = ContentBlock(text=OMITTED_IMAGE_TEXT)
        elif "toolResult" in block:
            result = block["toolResult"]
            block = ContentBlock(
                toolResult={
                    **result,
                    "content": [
                        {"text": OMITTED_IMAGE_TEXT} if "image" in item else item
                        for item in result["content"]
                    ],
                }
            )
        content.append(block)
    return Message(role=message["role"], content=content)


def _count_summary_usage(
    agent: StrandsAgent, summary_context: InvocationContext | None
) -> tuple[int, int]:
    """Add the summary call's tokens to the agent's metrics, and return them."""
    if summary_context is None:
        return 0, 0
    usage = summary_context.usage
    input_tokens, output_tokens = usage["inputTokens"], usage["outputTokens"]
    if input_tokens or output_tokens:
        agent.event_loop_metrics.update_usage(
            Usage(
                inputTokens=input_tokens,
                outputTokens=output_tokens,
                totalTokens=usage["totalTokens"],
            )
        )
    return input_tokens, output_tokens
