import enum
import json
from base64 import b64encode
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional, override
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import (
    Field,
    Relationship,
    SQLModel,
)

from database.keys.models import RobotKey
from database.tools.models import Tool
from s3 import S3Client
from templates.recovery_payload import RecoveryCapabilities, RecoveryPayload

if TYPE_CHECKING:
    from database.agent_graphs.models import AgentGraph


# A log section with the time it happened, to interleave sections chronologically.
type _TimestampedLog = tuple[datetime, str]

# Image uuid a logged message carries until its screenshot finishes uploading to S3.
PENDING_IMAGE_UUID = "<uuid_pending>"


async def _image_markdown(image: dict[str, Any]) -> str:
    image_uuid: str = image.get("uuid", "")
    if image_uuid == PENDING_IMAGE_UUID:
        return "_Image upload pending..._\n"
    image_b = await S3Client.download_bytes(image_uuid)
    if not image_b:
        return "_Image no longer available_\n"
    return f"![Image](data:image/jpeg;base64,{b64encode(image_b).decode()})\n"


async def _tool_result_markdown(tool_result: dict[str, Any]) -> str:
    log = ""
    for item in tool_result.get("content", []):
        if item.get("image", None):
            log += await _image_markdown(item["image"])
        else:
            log += f"{item}\n"
    return log


async def _content_part_markdown(part: dict[str, Any]) -> str:
    if "text" in part:
        return f"{part['text']}\n"
    if "image" in part:
        return await _image_markdown(part["image"])
    if "toolResult" in part:
        return await _tool_result_markdown(part["toolResult"])
    return f"{part}\n"  # Unknown dict format


def _message_timestamp(msg: dict[str, Any]) -> datetime:
    date = msg.get("timestamp", datetime.now(UTC))
    if isinstance(date, str):
        return datetime.strptime(date.replace(" ", "T"), "%Y-%m-%dT%H:%M:%S.%f%z")
    return date


async def _message_log(index: int, msg: dict[str, Any]) -> _TimestampedLog:
    log_msg = f"\n### Message {index}:\n"
    log_msg += f"- **{msg.get('role', '').capitalize()}:** "
    for part in msg.get("content", []):
        log_msg += await _content_part_markdown(part)
    return _message_timestamp(msg), log_msg


class AgentTrace(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    robot_exception_id: UUID | None = Field(
        foreign_key="robotexception.id", default=None
    )
    robot_exception: "RobotException" = Relationship(back_populates="agent_traces")

    agent_id: UUID = Field(foreign_key="agent.id")
    agent: "Agent" = Relationship(back_populates="traces")  # noqa: F821 # pyright:ignore[reportUndefinedVariable]

    # Grows with every GUI step, so access raises: read it with an explicit query.
    gui_traces: list["GUITrace"] = Relationship(
        back_populates="agent_trace",
        sa_relationship_kwargs={
            "lazy": "raise",
            "foreign_keys": "GUITrace.agent_trace_id",
            "single_parent": True,
        },
        cascade_delete=True,
    )

    sub_agents_traces: list["SubAgentTrace"] = Relationship(
        back_populates="parent_trace",
        sa_relationship_kwargs={"foreign_keys": "SubAgentTrace.parent_trace_id"},
    )

    tool_traces: list["ToolTrace"] = Relationship(back_populates="agent_trace")

    inputs: dict[str, Any] = Field(
        sa_type=JSONB,
    )

    # Input and output messages exchanged with the LLM
    # Each message is a dict with 'role' and 'content' keys
    messages: list[dict[str, Any]] | None = Field(default=None, sa_type=JSONB)

    cost: float = Field(default=0.0)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the trace was created.",
    )
    finished_at: datetime = Field(
        default=None,
        description="Timestamp of when the trace was closed.",
        nullable=True,
    )

    async def get_makdown_log(
        self, include_subtraces: bool = True, include_tool_traces: bool = True
    ) -> str:
        log = self._markdown_header()
        if not self.messages:
            return log + "_No messages recorded._\n"

        timestamped_logs = [
            *[await _message_log(i, msg) for i, msg in enumerate(self.messages)],
            *self._tool_trace_logs(include_tool_traces),
            *await self._sub_trace_logs(include_subtraces, include_tool_traces),
        ]
        timestamped_logs.sort(key=lambda x: x[0])
        return log + "".join(log_msg for _, log_msg in timestamped_logs)

    def _markdown_header(self) -> str:
        log = f"# Agent Trace: {self.id}\n"
        log += f"**Agent:** {self.agent.name} ({self.agent.id})\n\n"
        log += f"**Created At:** {self.created_at}\n\n"
        log += f"**Finished At:** {self.finished_at}\n\n"
        log += f"**Cost:** ${self.cost:.6f}\n\n"
        log += f"**Inputs:**\n```\n{json.dumps(self.inputs)}\n```\n\n"
        return log + "## Messages:\n"

    def _tool_trace_logs(self, include_details: bool) -> list[_TimestampedLog]:
        logs: list[_TimestampedLog] = []
        for i, tool_trace in enumerate(self.tool_traces):
            log_msg = f"\n### Tool Trace {i}:\n"
            if include_details:
                log_msg += f"- Tool: {tool_trace.tool.name} ({tool_trace.tool.id})\n"
                log_msg += f"  - Input: {tool_trace.input}\n"
                log_msg += f"  - Output: {tool_trace.output}\n"
                log_msg += f"  - Success: {tool_trace.success}\n"
            logs.append((tool_trace.created_at, log_msg))
        return logs

    async def _sub_trace_logs(
        self, include_subtraces: bool, include_tool_traces: bool
    ) -> list[_TimestampedLog]:
        logs: list[_TimestampedLog] = []
        for sub_trace in self.sub_agents_traces:
            child = sub_trace.child_trace
            log_msg = f"\n### Sub-Agent Trace: {child.agent.name}:\n"
            if include_subtraces:
                log_msg += "BEGIN Sub-Agent Trace:\n"
                log_msg += await child.get_makdown_log(
                    include_subtraces=include_subtraces,
                    include_tool_traces=include_tool_traces,
                )
                log_msg += "END Sub-Agent Trace:\n"
            logs.append((child.created_at, log_msg))
        return logs


class SubAgentTrace(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )
    parent_trace_id: UUID = Field(
        foreign_key="agenttrace.id",
        description="Foreign key to the parent agent trace.",
    )
    parent_trace: AgentTrace = Relationship(
        back_populates="sub_agents_traces",
        sa_relationship_kwargs={"foreign_keys": "SubAgentTrace.parent_trace_id"},
    )
    child_trace_id: UUID = Field(
        foreign_key="agenttrace.id",
        description="Foreign key to the child agent trace.",
    )
    child_trace: AgentTrace = Relationship(
        sa_relationship_kwargs={"foreign_keys": "SubAgentTrace.child_trace_id"},
    )

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        if self.parent_trace_id == self.child_trace_id:
            raise ValueError("A trace cannot be a sub-trace of itself.")


class RobotException(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    exception_details: dict[str, Any] = Field(
        sa_type=JSONB,
        nullable=True,
    )

    robot_key_id: UUID | None = Field(
        default=None,
        foreign_key="robotkey.id",
        description="RobotKey used to submit the exception",
    )
    robot_key: RobotKey | None = Relationship(
        sa_relationship_kwargs={
            "lazy": "noload",
        },
    )

    agent_traces: list[AgentTrace] = Relationship(back_populates="robot_exception")
    agent_graph_traces: list["AgentGraphTrace"] = Relationship(
        back_populates="robot_exception", cascade_delete=True
    )
    recovery_context: "RecoveryContext" = Relationship(back_populates="robot_exception")

    infered_success: bool = Field(
        default=False
    )  # Whether the agents reported a successful recovery
    reported_success: bool = Field(
        default=False
    )  # Whether the robot operator reported a successful recovery

    abort_code: str | None = Field(
        default=None,
        nullable=True,
        description="Error code the robot sent when it aborted the recovery.",
    )
    abort_description: str | None = Field(
        default=None,
        nullable=True,
        description="Reason the robot gave when it aborted the recovery.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the trace was created.",
    )
    finished_at: datetime = Field(
        default=None,
        description="Timestamp of when the trace was closed.",
        nullable=True,
    )


class RecoveryActivityFields(SQLModel):
    """The activity fields a UI log entry shares with the errored activity."""

    case_id: str = Field(default="")
    activity_id: str | None = Field(default=None)
    event_id: str = Field(default="")
    event_name: str | None = Field(default=None)
    model_event_id: str | None = Field(default=None)
    activity_name: str | None = Field(default=None)
    action_type: str = Field(default="")
    application: str | None = Field(default=None)
    input: str | None = Field(default=None)


class RecoveryUiLogEntry(RecoveryActivityFields, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    recovery_context_id: UUID | None = Field(
        default=None, foreign_key="recoverycontext.id"
    )
    recovery_context: "RecoveryContext" = Relationship(back_populates="ui_log_entries")

    position: int = Field(default=0)
    ui_element_target: str | None = Field(default=None)
    ui_group: str | None = Field(default=None)
    timestamp: str = Field(default="")
    previous_state: str | None = Field(default=None)
    current_state: str | None = Field(default=None)


class RecoveryErroredActivity(RecoveryActivityFields, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    recovery_context_id: UUID | None = Field(
        default=None,
        foreign_key="recoverycontext.id",
        unique=True,
    )
    recovery_context: "RecoveryContext" = Relationship(
        back_populates="errored_activity",
        sa_relationship_kwargs={
            "uselist": False,
        },
    )

    error_code: str = Field(default="")
    error_description: str = Field(default="")


class RecoveryContext(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    robot_exception_id: UUID | None = Field(
        default=None, foreign_key="robotexception.id"
    )
    robot_exception: RobotException | None = Relationship(
        sa_relationship_kwargs={"uselist": False}, back_populates="recovery_context"
    )

    task_name: str = Field(default="")
    platform: str = Field(default="")
    os: str = Field(default="")
    variables: dict[str, Any] = Field(default_factory=dict, sa_type=JSONB)
    model: str = Field(default="")
    capabilities: dict[str, Any] = Field(
        default_factory=lambda: RecoveryCapabilities().model_dump(), sa_type=JSONB
    )

    ui_log_entries: list[RecoveryUiLogEntry] = Relationship(
        back_populates="recovery_context",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "selectin",
            "order_by": "RecoveryUiLogEntry.position",
        },
        cascade_delete=True,
    )

    errored_activity: RecoveryErroredActivity | None = Relationship(
        back_populates="recovery_context",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "lazy": "joined",
            "uselist": False,
        },
        cascade_delete=True,
    )

    # TODO: Introduce RecoveryVariableMutation table when variable writebacks are enabled.

    @classmethod
    def from_payload(
        cls,
        *,
        payload: RecoveryPayload | dict[str, Any],
    ) -> "RecoveryContext":
        parsed_payload = (
            payload
            if isinstance(payload, RecoveryPayload)
            else RecoveryPayload.model_validate(payload)
        )

        recovery_context = cls(
            task_name=parsed_payload.task_name,
            platform=parsed_payload.platform,
            os=parsed_payload.os,
            variables=parsed_payload.variables,
            model=parsed_payload.model,
            capabilities=parsed_payload.capabilities.model_dump(),
        )

        for position, entry in enumerate(parsed_payload.ui_log):
            recovery_context.ui_log_entries.append(
                RecoveryUiLogEntry(position=position, **entry.model_dump())
            )

        recovery_context.errored_activity = RecoveryErroredActivity(
            **parsed_payload.errored_act.model_dump()
        )

        return recovery_context

    @override
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["ui_log"] = [entry.model_dump() for entry in self.ui_log_entries]
        if self.errored_activity:
            data["errored_act"] = self.errored_activity.model_dump()
        else:
            data["errored_act"] = None
        return data


class GUITrace(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    agent_trace_id: UUID = Field(
        foreign_key="agenttrace.id",
        ondelete="CASCADE",
    )
    agent_trace: AgentTrace = Relationship(
        sa_relationship_kwargs={"foreign_keys": "GUITrace.agent_trace_id"},
    )

    screenshot_key: str | None = Field(
        default=None
    )  # Screenshot name identifier in the configured S3 bucket

    action_type: str = Field(default="")  # e.g., "click", "input", etc.
    action_content: dict[str, Any] = Field(
        sa_type=JSONB,
        default={},
    )  # Details about the action performed

    success: bool = Field(default=False)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the gui interaction was started.",
    )
    finished_at: datetime = Field(
        default=None,
        description="Timestamp of when the gui interaction was finished.",
        nullable=True,
    )

    @staticmethod
    async def create(screenshot_b: bytes, **data: Any):
        screenshot_key = await S3Client.upload_bytes(
            screenshot_b,
            content_type="image/jpeg",
        )
        return GUITrace(screenshot_key=screenshot_key, **data)

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.agent_trace_id:
            raise ValueError("GUITrace must be associated with an AgentTrace.")


class ToolTrace(SQLModel, table=True):
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier",
        primary_key=True,
    )

    agent_trace_id: UUID = Field(foreign_key="agenttrace.id")
    agent_trace: AgentTrace = Relationship(
        back_populates="tool_traces",
        sa_relationship_kwargs={"foreign_keys": "ToolTrace.agent_trace_id"},
    )

    tool_id: UUID = Field(foreign_key="tool.id")
    tool: Tool = Relationship(back_populates="traces")

    input: dict[str, Any] = Field(
        sa_type=JSONB,
        default={},
    )

    output: str = Field(nullable=True, default=None)

    success: bool = Field(default=False)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the trace was created.",
    )
    finished_at: datetime = Field(
        default=None,
        description="Timestamp of when the trace was closed.",
        nullable=True,
    )


class AgentGraphTraceStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentGraphNodeTraceStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentGraphTrace(SQLModel, table=True):
    """One agent graph run of a recovery; its node traces are its steps, in order."""

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    agent_graph_id: UUID | None = Field(
        default=None, foreign_key="agentgraph.id", ondelete="SET NULL"
    )
    # A quoted ``X | None`` can't be resolved by SQLModel.
    agent_graph: Optional["AgentGraph"] = Relationship()
    robot_exception_id: UUID = Field(
        foreign_key="robotexception.id", ondelete="CASCADE", index=True
    )
    robot_exception: RobotException = Relationship(back_populates="agent_graph_traces")
    status: AgentGraphTraceStatus = Field(default=AgentGraphTraceStatus.RUNNING)
    failure_reason: str | None = Field(
        default=None,
        description="Why the run failed: the limit it hit, or the error it raised.",
    )
    terminal_node_key: str | None = Field(
        default=None, description="The key of the last node that completed."
    )
    answer: dict[str, Any] | None = Field(
        default=None, sa_type=JSONB, description="The answer sent to the robot."
    )
    node_traces: list["AgentGraphNodeTrace"] = Relationship(
        back_populates="agent_graph_trace",
        cascade_delete=True,
        sa_relationship_kwargs={
            "lazy": "selectin",
            "order_by": "AgentGraphNodeTrace.execution_index",
        },
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = Field(default=None)

    async def get_markdown_log(self) -> str:
        """The run's header, then each node execution followed by its agent trace."""
        graph = self.agent_graph
        log = f"# Agent Graph Trace: {self.id}\n\n"
        log += (
            f"**Agent Graph:** {graph.name} ({graph.id})\n\n"
            if graph is not None
            else "**Agent Graph:** deleted\n\n"
        )
        log += f"**Robot Exception:** {self.robot_exception_id}\n\n"
        log += f"**Status:** {self.status.value}\n\n"
        if self.failure_reason is not None:
            log += f"**Failure Reason:** {self.failure_reason}\n\n"
        if self.terminal_node_key is not None:
            log += f"**Terminal Node:** {self.terminal_node_key}\n\n"
        log += f"**Created At:** {self.created_at}\n\n"
        log += f"**Finished At:** {self.finished_at}\n\n"
        log += (
            f"**Answer:**\n```json\n{json.dumps(self.answer, indent=2)}\n```\n"
            if self.answer is not None
            else "**Answer:** none\n"
        )
        for node_trace in self.node_traces:
            log += "\n" + await node_trace.get_markdown_log()
        return log


class AgentGraphNodeTrace(SQLModel, table=True):
    """One node execution of an agent graph run, with how the run got there."""

    __table_args__ = (UniqueConstraint("agent_graph_trace_id", "execution_index"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    agent_graph_trace_id: UUID = Field(
        foreign_key="agentgraphtrace.id", ondelete="CASCADE", index=True
    )
    agent_graph_trace: AgentGraphTrace = Relationship(back_populates="node_traces")
    execution_index: int = Field(description="The node's place in the run, from 0.")
    node_key: str
    agent_trace_id: UUID | None = Field(
        default=None, foreign_key="agenttrace.id", ondelete="SET NULL"
    )
    agent_trace: AgentTrace | None = Relationship()
    entry_edge_id: UUID | None = Field(
        default=None,
        foreign_key="agentgraphedge.id",
        ondelete="SET NULL",
        description="The edge the run entered the node by; null for the entry node.",
    )
    entry_condition: dict[str, Any] | None = Field(
        default=None,
        sa_type=JSONB,
        description="The entry edge's condition when the node ran; null if unconditional.",
    )
    status: AgentGraphNodeTraceStatus = Field(default=AgentGraphNodeTraceStatus.RUNNING)
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = Field(default=None)

    async def get_markdown_log(self) -> str:
        """How the run entered the node and how it ended, then its agent trace."""
        agent_trace = self.agent_trace
        log = f"## Node {self.execution_index}: {self.node_key}\n\n"
        log += (
            f"**Agent:** {agent_trace.agent.name} ({agent_trace.agent_id})\n\n"
            if agent_trace is not None
            else "**Agent:** unknown\n\n"
        )
        if self.execution_index == 0:
            log += "**Entered By:** the entry node\n\n**Condition:** none\n\n"
        else:
            log += (
                f"**Entered By:** edge {self.entry_edge_id}\n\n"
                if self.entry_edge_id is not None
                else "**Entered By:** a deleted edge\n\n"
            )
            log += (
                f"**Condition:** `{json.dumps(self.entry_condition)}`\n\n"
                if self.entry_condition is not None
                else "**Condition:** unconditional\n\n"
            )
        log += f"**Status:** {self.status.value}\n\n"
        if self.error is not None:
            log += f"**Error:** {self.error}\n\n"
        if agent_trace is None:
            return log + "_No agent trace recorded._\n"
        return log + await agent_trace.get_makdown_log()


class AgentGraphNodeTraceRead(SQLModel):
    id: UUID
    execution_index: int
    node_key: str
    agent_trace_id: UUID | None
    entry_edge_id: UUID | None
    entry_condition: dict[str, Any] | None
    status: AgentGraphNodeTraceStatus
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class AgentGraphTraceRead(SQLModel):
    """An agent graph run with its node traces in execution order."""

    id: UUID
    agent_graph_id: UUID | None
    robot_exception_id: UUID
    status: AgentGraphTraceStatus
    failure_reason: str | None
    terminal_node_key: str | None
    answer: dict[str, Any] | None
    node_traces: list[AgentGraphNodeTraceRead]
    created_at: datetime
    finished_at: datetime | None
