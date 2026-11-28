from collections.abc import Iterable, Iterator
from copy import deepcopy
from datetime import UTC, datetime
from threading import Lock
from typing import Any, override
from uuid import UUID, uuid4

from opentelemetry import trace as trace_api
from pydantic import ValidationError
from strands.hooks import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.types.content import ContentBlock, Message
from strands.types.tools import ToolResultContent

from database.agent_graphs.runner import AGENT_GRAPH_CONTEXT_KEY, AgentGraphContext
from database.engine import open_session, update_by_id, update_row
from database.logging.models import (
    PENDING_IMAGE_UUID,
    AgentGraphNodeTrace,
    AgentTrace,
    GUITrace,
    SubAgentTrace,
    ToolTrace,
)
from observability import attributes as attrs
from settings import MAX_GUI_IMAGES

MAX_REQUIRED_TOOL_REMINDERS = 3

OMITTED_IMAGE_TEXT = "ommited screenshot due to max allowed images per message"


class MaxGuiImagesHook(HookProvider):
    """Keep only the most recent images in the conversation before each model call."""

    def __init__(self, max_images: int = MAX_GUI_IMAGES):
        """
        Initializer.

        Args:
            max_images: Maximum number of screenshots to keep. Older images are
                replaced with a text placeholder.
        """
        if max_images < 0:
            raise ValueError("max_images must be non-negative")
        self.max_images = max_images

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self.limit_images)

    def limit_images(self, event: BeforeModelCallEvent) -> None:
        remaining = self.max_images
        for message in reversed(event.agent.messages):
            remaining = self._limit_message_images(message, remaining)

    def _limit_message_images(self, message: Message, remaining: int) -> int:
        content = message["content"]
        for index in range(len(content) - 1, -1, -1):
            block = content[index]
            if "image" in block:
                remaining = self._handle_image_block(block, remaining)
            elif "toolResult" in block:
                result_content = block["toolResult"].get("content", [])
                for result_index in range(len(result_content) - 1, -1, -1):
                    result_block = result_content[result_index]
                    if "image" in result_block:
                        remaining = self._handle_image_block(result_block, remaining)
        return remaining

    @staticmethod
    def _handle_image_block(
        block: ContentBlock | ToolResultContent, remaining: int
    ) -> int:
        if remaining > 0:
            return remaining - 1

        _ = block.pop("image", None)
        block["text"] = OMITTED_IMAGE_TEXT
        return remaining


class RequireToolCalls(HookProvider):
    """Sends the agent back until each required tool has been called successfully.

    After at most `max_reminders` reminders the agent may finish anyway.
    """

    def __init__(
        self,
        required_tools: Iterable[str],
        max_reminders: int = MAX_REQUIRED_TOOL_REMINDERS,
    ):
        self.required_tools = set(required_tools)
        self.max_reminders = max_reminders
        self.called: set[str] = set()
        self.reminders = 0

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        if not self.required_tools:
            return
        registry.add_callback(AfterToolCallEvent, self.record_call)
        registry.add_callback(AfterInvocationEvent, self.remind)

    def record_call(self, event: AfterToolCallEvent) -> None:
        if event.exception is None and event.result.get("status") != "error":
            self.called.add(event.tool_use["name"])

    def remind(self, event: AfterInvocationEvent) -> None:
        missing = sorted(self.required_tools - self.called)
        if not missing or self.reminders >= self.max_reminders:
            return
        self.reminders += 1
        names = ", ".join(f"`{name}`" for name in missing)
        event.resume = (
            f"You haven't called the required tool(s) {names} successfully. Call them "
            "before finishing; if a call is rejected, correct it as its error says and "
            "call it again."
        )


class LimitToolCounts(HookProvider):
    """Limits the number of times tools can be called per agent invocation"""

    def __init__(self, max_tool_counts: dict[str, int]):
        """
        Initializer.

        Args:
            max_tool_counts: A dictionary mapping tool names to max call counts for
                tools. If a tool is not specified in it, the tool can be called as many
                times as desired
        """
        self.max_tool_counts = max_tool_counts
        self.tool_counts = {}
        self._lock = Lock()

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(BeforeToolCallEvent, self.intercept_tool)
        registry.add_callback(AfterToolCallEvent, self.intercept_response)

    def reset_counts(
        self,
        event: BeforeInvocationEvent,  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        with self._lock:
            self.tool_counts = {}

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        with self._lock:
            max_tool_count = self.max_tool_counts.get(tool_name)
            tool_count = self.tool_counts.get(tool_name, 0) + 1
            self.tool_counts[tool_name] = tool_count

        if max_tool_count and tool_count > max_tool_count:
            event.cancel_tool = (
                f"Tool '{tool_name}' has been invoked too many and is now being throttled. "
                f"DO NOT CALL THIS TOOL ANYMORE "
            )

    def intercept_response(self, event: AfterToolCallEvent):
        # Remove one call if validation exception happened
        if _is_argument_error(event):
            tool_name = event.tool_use["name"]
            with self._lock:
                tool_count = self.tool_counts.get(tool_name, 0) - 1
                self.tool_counts[tool_name] = max(0, tool_count)


def _is_argument_error(event: AfterToolCallEvent) -> bool:
    """Whether the tool call failed on its arguments rather than in the tool."""
    if isinstance(event.exception, (ValidationError, TypeError)):
        return True
    if not event.result or event.result.get("status", "") != "error":
        return False
    text = next(iter(event.result.get("content", [])), {}).get("text", "")
    return "TypeError" in text or "ValidationError" in text


class ToolLoggingHook(HookProvider):
    def __init__(self, agent_trace_id: UUID, tools: dict[str, UUID]):
        """Initializes the ToolLoggingHook.

        Args:
            agent_trace_id: The UUID of the agent trace.
            tools: A dictionary mapping tool names to be logged to their UUIDs.
        """
        self.agent_trace_id = agent_trace_id
        self.tools = tools
        self.trace_id: UUID

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeToolCallEvent, self.log_tool_call)
        registry.add_callback(AfterToolCallEvent, self.log_tool_response)

    def log_tool_call(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "")
        if tool_name not in self.tools:
            return

        with open_session() as session:
            trace_id = uuid4()
            tool_trace = ToolTrace(
                id=trace_id,
                agent_trace_id=self.agent_trace_id,
                tool_id=self.tools[tool_name],
                input=event.tool_use.get("input", ""),
            )
            session.add(tool_trace)
            session.commit()
            self.trace_id = trace_id

    def log_tool_response(self, event: AfterToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "")
        if tool_name not in self.tools:
            return

        if event.exception:
            output, success = str(event.exception), False
        else:
            # FIXME: Support only text output for now
            output = next(iter(event.result.get("content", [])), {}).get("text", "")
            success = True

        update_by_id(
            ToolTrace,
            self.trace_id,
            output=output,
            success=success,
            finished_at=datetime.now(UTC),
        )


class AgentLoggingHook(HookProvider):
    def __init__(
        self,
        agent_id: UUID,
        invocation_state: dict[str, Any],
        parent_trace_id: UUID | None = None,
    ):
        """Initializes the AgentLoggingHook.

        Args:
            agent_id: The UUID of the agent.
            invocation_state: The state of the agent invocation.
            parent_agent_id: The UUID of the parent agent, if any.
        """
        self.parent_trace_id = parent_trace_id
        self.agent_id = agent_id
        self.invocation_state = invocation_state
        # We generate here the id for the agent trace so we can reference it outside
        self.agent_trace_id = uuid4()
        self.created = False
        self.messages: list[dict[str, Any]] = []
        self.created: bool = False

    @override
    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self.log_start)
        registry.add_callback(MessageAddedEvent, self.log_message)

    def log_start(self, event: BeforeInvocationEvent) -> None:  # pyright: ignore[reportUnusedParameter]
        # Skip if already logged, can happen if multiple invocations are made
        if self.created:
            return
        node_trace_id = self._claim_open_node_trace()
        with open_session() as session:
            # We first register the agent trace here
            trace = AgentTrace(
                id=self.agent_trace_id,
                agent_id=self.agent_id,
                robot_exception_id=self.invocation_state.get(
                    "robot_exception_id", None
                ),
                inputs=self.invocation_state.get("inputs", {}),
            )
            session.add(trace)

            if self.parent_trace_id:
                sub_trace = SubAgentTrace(
                    parent_trace_id=self.parent_trace_id,
                    child_trace_id=self.agent_trace_id,
                )
                session.add(sub_trace)

            if node_trace_id is not None:
                # The agent trace is inserted first: the node trace links to it.
                session.flush()
                _ = update_row(
                    session,
                    AgentGraphNodeTrace,
                    node_trace_id,
                    agent_trace_id=self.agent_trace_id,
                )

            session.commit()
            self.created = True
            # BeforeInvocationEvent runs inside the Strands agent span: link the
            # span to the AgentTrace row so traces and the audit tables can be joined.
            trace_api.get_current_span().set_attributes(
                {
                    attrs.AGENT_ID: str(self.agent_id),
                    attrs.AGENT_TRACE_ID: str(self.agent_trace_id),
                }
            )

    def _claim_open_node_trace(self) -> UUID | None:
        """The open node trace of a traced agent graph run, if this agent is its node.

        A node's agent trace is a root; sub-agents of the node have a parent.
        """
        context = self.invocation_state.get(AGENT_GRAPH_CONTEXT_KEY)
        if (
            self.parent_trace_id is not None
            or not isinstance(context, AgentGraphContext)
            or context.trace is None
        ):
            return None
        return context.trace.claim_open_node()

    def log_message(self, event: MessageAddedEvent) -> None:
        message: dict[str, Any] = {
            "role": event.message.get("role", "unknown"),
            # Includes text, toolcalls, images, etc.
            "content": deepcopy(event.message.get("content", [])),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        for part in message["content"]:
            for block in _image_blocks(part):
                # Drop the possibly large image data; the S3 uuid is filled in later.
                block["image"] = {"uuid": PENDING_IMAGE_UUID}

        self.messages.append(message)
        self.update_trace()

    def update_trace(self, finished: bool = False, cost: float = 0.0):
        """Write the messages logged so far, and the finish, in one UPDATE.

        The whole list is written each time, not appended: an earlier message
        changes when its screenshot's S3 uuid is filled in, and Postgres rewrites
        the whole JSONB value on any change anyway (see TECH_DEBT.md).
        """
        values: dict[str, object] = {"messages": self.messages}
        if finished:
            values |= {"finished_at": datetime.now(UTC), "cost": cost}
        update_by_id(AgentTrace, self.agent_trace_id, **values)

    async def register_gui_trace(
        self,
        action_type: str,
        action_content: dict[str, Any],
        screenshot_bytes: bytes,
        started_at: datetime,
        finished_at: datetime,
        success: bool = True,
    ):
        gui_trace = await GUITrace.create(
            screenshot_b=screenshot_bytes,
            agent_trace_id=self.agent_trace_id,
            action_type=action_type,
            action_content=action_content,
            success=success,
            created_at=started_at,
            finished_at=finished_at,
        )
        with open_session(expire_on_commit=False) as session:
            session.add(gui_trace)
            session.commit()

        # Messages are logged before their screenshot's GUI trace is saved, so the
        # latest image still pending gets the S3 uuid of the screenshot now.
        if self._fill_pending_image(str(gui_trace.screenshot_key)):
            self.update_trace()

    def _fill_pending_image(self, image_uuid: str) -> bool:
        """Give the latest logged image still pending `image_uuid`; whether there was one."""
        for msg in reversed(self.messages):
            for part in msg["content"]:
                for block in _image_blocks(part):
                    if block["image"].get("uuid") == PENDING_IMAGE_UUID:
                        block["image"]["uuid"] = image_uuid
                        return True
        return False


def _image_blocks(part: object) -> Iterator[dict[str, Any]]:
    """The blocks of a logged message part that hold an image.

    That is the part itself, or the items of the tool result it carries.
    """
    if not isinstance(part, dict):
        return
    if part.get("image"):
        yield part
    elif (tool_result := part.get("toolResult")) and (
        content := tool_result.get("content")
    ):
        yield from (item for item in content if item.get("image"))
