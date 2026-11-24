"""Telemetry for an agent's context compression (``agent_tools/context_compression.py``).

A summary that replaced old messages is an event on the current span, the
event-loop cycle whose model call it made room for. A summary that failed is a
recovered failure, so it is a WARN log event: the run goes on without it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace

from observability import attributes as attrs
from observability.events import brief, error_type, log_event

if TYPE_CHECKING:
    from agent_tools.context_compression import ContextCompression

logger = logging.getLogger(__name__)


def context_compressed(compression: ContextCompression) -> None:
    """Old messages were replaced by a summary before a model call."""
    attributes = {
        attrs.AGENT_CONTEXT_COMPRESSION_TRIGGER: compression.trigger,
        attrs.AGENT_CONTEXT_COMPRESSION_MESSAGES_REPLACED: compression.messages_replaced,
        attrs.AGENT_CONTEXT_COMPRESSION_TOKENS_BEFORE: compression.tokens_before,
        attrs.AGENT_CONTEXT_COMPRESSION_TOKENS_AFTER: compression.tokens_after,
        attrs.GEN_AI_USAGE_INPUT_TOKENS: compression.input_tokens,
        attrs.GEN_AI_USAGE_OUTPUT_TOKENS: compression.output_tokens,
    }
    trace.get_current_span().add_event(
        attrs.EVENT_AGENT_CONTEXT_COMPRESSED,
        {key: value for key, value in attributes.items() if value is not None},
    )


def context_compression_failed(trigger: str, error: BaseException | None) -> None:
    """The context was due for a summary, but none replaced the old messages."""
    log_event(
        logger,
        attrs.EVENT_AGENT_CONTEXT_COMPRESSION_FAILED,
        {
            attrs.AGENT_CONTEXT_COMPRESSION_TRIGGER: trigger,
            attrs.ERROR_TYPE: error_type(error),
            attrs.EXCEPTION_MESSAGE: brief(error) if error is not None else None,
        },
        logging.WARNING,
    )
