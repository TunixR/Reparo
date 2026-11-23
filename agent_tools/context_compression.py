"""Context compression for recovery agents: old messages become a summary, and each one is reported.

``ReportingSummarizingManager`` is Strands' ``SummarizingConversationManager``
with a recovery-aware summary prompt and three additions Strands leaves out:

- **Reporting.** Strands inserts the summary into ``agent.messages`` by slice
  assignment and fires no hook event, so it is invisible to the trace. Each
  summary is reported to ``on_compression`` (the agent's ``AgentLoggingHook``)
  and as a span event; a failed one as a WARN log event.
- **Usage.** The summary is a model call outside the agent loop. Strands adds
  its tokens to the request's ``InvocationContext`` but not to
  ``event_loop_metrics``, which Reparo prices an agent from; they are added there.
- **No blocking.** Strands' proactive hook is synchronous and makes the summary
  call on a worker thread it waits for, which stalls the event loop, and every
  recovery and robot socket on it, for the length of the call. Proactive
  compression runs from an async hook instead, with the same 70% threshold,
  and waits for the summary on a thread.

Overflow recovery (``reduce_context`` with an error) is still Strands' own
synchronous call; see TECH_DEBT.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, override

from strands.agent.conversation_manager import SummarizingConversationManager
from strands.agent.conversation_manager.conversation_manager import (
    DEFAULT_COMPRESSION_THRESHOLD,
)
from strands.agent.invocation_context import InvocationContext
from strands.hooks import BeforeModelCallEvent, HookRegistry
from strands.types.content import Message, Messages
from strands.types.event_loop import Usage

from observability.instrumentation.context_compression import (
    context_compressed,
    context_compression_failed,
)

if TYPE_CHECKING:
    from strands import Agent as StrandsAgent
    from strands.models import Model

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
    """The model call's projected input tokens, as Strands estimated them."""
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
        summary_ratio: Share of the history a summary replaces.
        on_compression: Called with each summary that replaced messages.
    """

    def __init__(
        self,
        *,
        pin_first: int,
        preserve_recent_messages: int = 10,
        summary_ratio: float = SUMMARY_RATIO,
        on_compression: Callable[[ContextCompression], None] | None = None,
    ) -> None:
        # Proactive compression is this class's own async hook; Strands' is left off.
        super().__init__(
            summary_ratio=summary_ratio,
            preserve_recent_messages=preserve_recent_messages,
            summarization_system_prompt=RECOVERY_SUMMARIZATION_PROMPT,
            pin_first=pin_first,
            proactive_compression=None,
        )
        self.compression_threshold = DEFAULT_COMPRESSION_THRESHOLD
        self.on_compression = on_compression
        self.summary_message: Message | None = None
        """The latest summary in the history, if any."""
        self._projected_input_tokens: int | None = None
        self._summary_error: Exception | None = None

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        super().register_hooks(registry, **kwargs)
        registry.add_callback(BeforeModelCallEvent, self._compress_before_model_call)

    async def _compress_before_model_call(self, event: BeforeModelCallEvent) -> None:
        tokens = event.projected_input_tokens
        self._projected_input_tokens = tokens
        if (
            tokens is None
            or event.agent.model.estimate_utilization(tokens)
            < self.compression_threshold
        ):
            return
        try:
            await asyncio.to_thread(
                self.reduce_context,
                event.agent,
                invocation_context=event.invocation_context,
            )
        except Exception as exc:
            # Best effort, like Strands' own proactive compression: the call goes ahead.
            context_compression_failed("proactive", exc)

    @override
    def reduce_context(
        self, agent: StrandsAgent, e: Exception | None = None, **kwargs: Any
    ) -> None:
        trigger: CompressionTrigger = "proactive" if e is None else "overflow"
        context = kwargs.get("invocation_context")
        used_before = _tokens_used(context)
        before = list(agent.messages)
        self._summary_error = None
        try:
            super().reduce_context(agent, e, **kwargs)
        except Exception as exc:
            _ = _count_summary_usage(agent, context, used_before)
            context_compression_failed(trigger, exc)
            raise
        input_tokens, output_tokens = _count_summary_usage(agent, context, used_before)

        kept = {id(message) for message in agent.messages}
        known = {id(message) for message in before}
        summary = next((m for m in agent.messages if id(m) not in known), None)
        if summary is None:
            # A proactive summary failed (Strands swallows it), or an overflow fell
            # back to truncating text blocks.
            context_compression_failed(trigger, self._summary_error)
            return
        self.summary_message = summary
        replaced = [message for message in before if id(message) not in kept]
        tokens_before = self._projected_input_tokens
        if tokens_before is None:
            tokens_before = _estimate_tokens(agent.model, before)
        removed = _estimate_tokens(agent.model, replaced)
        added = _estimate_tokens(agent.model, [summary])
        tokens_after = (
            None
            if tokens_before is None or removed is None or added is None
            else tokens_before - removed + added
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
        context_compressed(
            trigger=trigger,
            messages_replaced=compression.messages_replaced,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
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


def _tokens_used(context: object) -> tuple[int, int]:
    if not isinstance(context, InvocationContext):
        return 0, 0
    return context.usage["inputTokens"], context.usage["outputTokens"]


def _count_summary_usage(
    agent: StrandsAgent, context: object, used_before: tuple[int, int]
) -> tuple[int, int]:
    """Add the summary call's tokens to the agent's metrics, and return them.

    Strands adds them to the request's ``InvocationContext`` only, so the
    difference since ``used_before`` is the summary call.
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


def _estimate_tokens(model: Model, messages: Messages) -> int | None:
    """The model's estimate of ``messages``, or ``None`` if it can't give one.

    ``count_tokens`` is async and this runs inside the agent's event loop, or on a
    thread without one, so it gets a loop of its own on a thread.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda: asyncio.run(model.count_tokens(messages))
            ).result()
    except Exception:
        return None
