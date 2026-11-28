"""The agent tables and their API schemas.

How an agent runs depends on its ``type`` and lives in `database.agents.behaviours`;
this module holds no agent loop and imports nothing from there.
"""

import enum
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, override

from pydantic import BaseModel, TypeAdapter, ValidationError, create_model
from sqlalchemy import CheckConstraint, Column, Dialect, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Enum, Field, Relationship, SQLModel
from strands.models import Model

from database.agents.specs import (
    FIELD_SPECS,
    FieldSpec,
    FieldSpecs,
    OutputModelSpec,
    build_input_model,
)
from database.provider.models import Router
from database.tools.models import Tool

# Module level: inside `Agent` the name `type` is the agent type field.
type ModelClass = type[BaseModel]


OUTPUT_MODEL_SPEC: TypeAdapter[OutputModelSpec | None] = TypeAdapter(
    OutputModelSpec | None
)

_SPEC_COLUMNS: dict[str, TypeAdapter[Any]] = {
    "arguments": FIELD_SPECS,
    "output_model": OUTPUT_MODEL_SPEC,
}
"""The `Agent` columns that hold definitions, and the adapter that validates each."""


class SpecJSON(TypeDecorator[Any]):
    """A JSONB column that stores, and always loads, a valid definition.

    `Agent` validates a definition when it is assigned, so binding one only dumps it;
    a raw value written past the model (a bulk update) is validated here instead.
    """

    impl = JSONB
    cache_ok = True

    def __init__(self, adapter: TypeAdapter[Any]) -> None:
        super().__init__()
        self.adapter: TypeAdapter[Any] = adapter

    @override
    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        return self.adapter.dump_python(
            self.adapter.validate_python(value), mode="json"
        )

    @override
    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        return None if value is None else self.adapter.validate_python(value)


def _as_json(value: Any) -> Any:
    """``value`` with its pydantic models dumped to JSON, so they validate as data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_as_json(item) for item in value]  # pyright: ignore[reportUnknownVariableType] - JSON
    return value


class AgentType(str, enum.Enum):
    """How an agent runs; `database.agents.behaviours` maps each type to its class."""

    Agent = "Agent"
    TooledGuiAgent = "TooledGuiAgent"
    TextGuiAgent = "TextGuiAgent"


class Agent(SQLModel, table=True):
    class InputType(str, enum.Enum):
        TEXT = "TEXT"
        IMAGETEXT = "IMAGETEXT"

    class ReasoningEffort(str, enum.Enum):
        """Reasoning effort sent as ``reasoning_effort``; labels vary by model."""

        MINIMAL = "minimal"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        XHIGH = "xhigh"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the agent.",
        primary_key=True,
    )
    name: str = Field(..., description="Name of the agent.")
    description: str = Field(..., description="Description of the agent.")
    prompt: str = Field(..., description="The prompt used to initialize the agent.")
    output_model: OutputModelSpec | None = Field(
        default=None,
        sa_type=SpecJSON(OUTPUT_MODEL_SPEC),
        description=(
            "The agent's structured output; null for none. A GUI agent always "
            "reports a GuiRunReport."
        ),
    )
    # TEXT, IMAGETEXT
    input_type: InputType = Field(
        ...,
        sa_column=Column(
            Enum(InputType),
            nullable=False,
            default=InputType.TEXT,
        ),
        description="The type of input the agent can process.",
    )
    enabled: bool = Field(
        True, description="Indicates whether the agent is enabled or not."
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        sa_column=Column(Enum(ReasoningEffort), nullable=True),
        description=(
            "Reasoning effort requested from reasoning models. Unset leaves the "
            "provider default, which can be the highest effort."
        ),
    )

    arguments: list[FieldSpec] = Field(
        default_factory=list,
        sa_column=Column(SpecJSON(FIELD_SPECS), nullable=False, server_default="[]"),
        description="The agent's inputs, in the order its instruction lists them.",
    )

    router_id: uuid.UUID = Field(
        foreign_key="router.id",
        description="Foreign key to the router used by the agent.",
    )
    router: Router = Relationship(
        back_populates="agents",
        sa_relationship_kwargs={"lazy": "joined"},
    )

    tools: list["AgentTool"] = Relationship(
        back_populates="agent",
        sa_relationship_kwargs={"lazy": "joined"},
    )
    sub_agents: list["SubAgent"] = Relationship(
        back_populates="parent_agent",
        sa_relationship_kwargs={
            "lazy": "joined",
            "foreign_keys": "SubAgent.parent_agent_id",
        },
    )

    graph_nodes: list["AgentGraphNode"] = Relationship(  # noqa: F821 # pyright:ignore[reportUndefinedVariable]
        back_populates="agent"
    )

    # Grows with every run, so access raises: read it with an explicit query.
    traces: list["AgentTrace"] = Relationship(  # noqa: F821 # pyright:ignore[reportUndefinedVariable]
        back_populates="agent",
        sa_relationship_kwargs={"lazy": "raise"},
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the agent was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the agent was last updated.",
    )

    type: AgentType = Field(
        ...,
        sa_column=Column(
            Enum(AgentType),
            nullable=False,
            default=AgentType.Agent,
        ),
        description="The type of the agent, which decides how it runs.",
    )
    parser_path: str | None = Field(
        default=None,
        description=(
            "The parser that turns a TextGuiAgent's replies into actions, as "
            "'module:attribute'; null for every other type."
        ),
    )

    def get_tools(self) -> list[Tool]:
        """Return the list of tools associated with the agent."""
        return [at.tool for at in self.tools]

    def get_sub_agents(self) -> list["Agent"]:
        """Return the list of sub-agents associated with the agent."""
        return [sa.child_agent for sa in self.sub_agents if sa.child_agent.enabled]

    def get_model_params(self) -> dict[str, Any]:
        """Return the request parameters this agent sends to its router's model."""
        params: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.ReasoningEffort(
                self.reasoning_effort
            ).value
        return params

    def get_model(self) -> Model:
        """Return the router's model configured with this agent's parameters."""
        return self.router.get_model(**self.get_model_params())

    def __init__(self, **data: Any) -> None:
        # Typed loosely: definitions may come as JSON, which __setattr__ validates,
        # and the type checker would otherwise ask for every defaulted column. No
        # rule across fields runs here, since model_validate() builds an empty
        # instance first; `validate_agent` runs them (database.agents.behaviours).
        super().__init__(**data)

    @override
    def __setattr__(self, name: str, value: Any) -> None:
        # Validate here, not with @validates: SQLModel stores the assigned value over
        # what a validator returns, and these columns must always hold definitions.
        # Construction assigns through here too.
        if adapter := _SPEC_COLUMNS.get(name):
            try:
                value = adapter.validate_python(_as_json(value))
            except ValidationError as exc:
                raise ValueError(f"Invalid {name}: {exc}") from exc
        super().__setattr__(name, value)

    def input_class(self) -> ModelClass:
        """Return the model that validates this agent's inputs (see `validate_input`)."""
        return build_input_model(self.arguments)

    def get_input_schema(self) -> dict[str, Any]:
        """Return the tool input schema of this agent: its arguments as a JSON Schema."""
        schema = self.input_class().model_json_schema()
        _ = schema.pop("title", None)
        return {"json": schema}

    def validate_input(self, **kwargs: Any) -> dict[str, Any]:
        """Validate inputs against this agent's arguments and return them with defaults.

        Validation is strict: a bool is not an int and a string is not a number, but
        an int is a float. None is a value, valid where the type allows it. Models,
        such as those a graph context holds, are validated as their JSON.
        """
        values = {name: _as_json(value) for name, value in kwargs.items()}
        try:
            return self.input_class().model_validate(values).model_dump()
        except ValidationError as exc:
            raise ValueError("; ".join(_input_errors(exc))) from exc

    @classmethod
    def required_capabilities_for(
        cls, input_type: "Agent.InputType"
    ) -> set[Router.Capability]:
        """Return the router capabilities required by an agent input type."""
        normalized = cls.InputType(input_type)
        if normalized == cls.InputType.IMAGETEXT:
            return {Router.Capability.TEXT, Router.Capability.VISION}
        return {Router.Capability.TEXT}

    @classmethod
    def validate_capabilities_support_input_type(
        cls,
        capabilities: "list[Router.Capability]",
        input_type: "Agent.InputType",
        *,
        router_name: str = "router",
    ) -> None:
        """Raise when the given router capabilities cannot support the input type."""
        normalized = cls.InputType(input_type)
        available = {Router.Capability(capability) for capability in capabilities}
        missing = sorted(
            capability.value
            for capability in cls.required_capabilities_for(normalized)
            if capability not in available
        )
        if missing:
            available_values = sorted(capability.value for capability in available)
            raise ValueError(
                f"Agent input type '{normalized.value}' requires router capabilities \
                {missing}, but router '{router_name}' only supports {available_values}."
            )

    def apply_update(self, payload: "AgentUpdate", router: Router) -> None:
        """Apply an update payload.

        It checks no rule across fields: `validate_agent` does, once the update is
        applied (`database.agents.behaviours`).
        """
        data = {
            key: value
            for key, value in payload.model_dump(exclude_unset=True).items()
            if value is not None
        }

        # An explicit null clears the effort back to the provider default, the
        # output model to none, and the parser path.
        for key in ("reasoning_effort", "output_model", "parser_path"):
            if key in payload.model_fields_set:
                data[key] = getattr(payload, key)

        for key, value in data.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                raise ValueError(f"Failed to set field '{key}': {e!s}") from e

        self.router = router
        self.router_id = router.id


GUI_IGNORED_NOTE = (
    " A GUI agent (TooledGuiAgent, TextGuiAgent) ignores it; create and update "
    "list it in ignored_fields."
)


class AgentCreate(SQLModel):
    """Schema for creating a new agent."""

    name: str = Field(description="Name of the agent.")
    description: str = Field(description="Description of the agent.")
    prompt: str = Field(description="The prompt used to initialize the agent.")
    arguments: FieldSpecs = Field(
        default_factory=list,
        description=(
            "The agent's inputs, in the order its instruction lists them. A GUI "
            "agent's are replaced by the fixed GUI inputs."
        ),
    )
    output_model: OutputModelSpec | None = Field(
        default=None,
        description="The agent's structured output." + GUI_IGNORED_NOTE,
    )
    input_type: Agent.InputType = Field(
        default=Agent.InputType.TEXT,
        description=(
            "The type of input the agent can process. A GUI agent's is always "
            "IMAGETEXT."
        ),
    )
    enabled: bool = Field(default=True, description="Whether the agent is enabled.")
    reasoning_effort: Agent.ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort for reasoning models; unset keeps the provider default.",
    )
    router_id: uuid.UUID = Field(
        description="Foreign key to the router used by the agent."
    )
    type: AgentType = Field(
        default=AgentType.Agent,
        description="The type of the agent, which decides how it runs.",
    )
    parser_path: str | None = Field(
        default=None,
        description=(
            "Required for a TextGuiAgent and rejected for every other type: the "
            "parser of its replies, as 'module:attribute'."
        ),
    )


class AgentUpdate(SQLModel):
    """Schema for updating an existing agent."""

    name: str | None = Field(default=None, description="Name of the agent.")
    description: str | None = Field(
        default=None, description="Description of the agent."
    )
    prompt: str | None = Field(
        default=None, description="The prompt used to initialize the agent."
    )
    arguments: FieldSpecs | None = Field(
        default=None,
        description=(
            "The agent's inputs, in order; replaces the current list. A GUI "
            "agent's are replaced by the fixed GUI inputs."
        ),
    )
    output_model: OutputModelSpec | None = Field(
        default=None,
        description="The agent's structured output; null removes it."
        + GUI_IGNORED_NOTE,
    )
    input_type: Agent.InputType | None = Field(
        default=None,
        description=(
            "The type of input the agent can process. A GUI agent's is always "
            "IMAGETEXT."
        ),
    )
    enabled: bool | None = Field(
        default=None, description="Whether the agent is enabled."
    )
    reasoning_effort: Agent.ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort for reasoning models; null clears it.",
    )
    router_id: uuid.UUID | None = Field(
        default=None, description="Foreign key to the router used by the agent."
    )
    type: AgentType | None = Field(
        default=None, description="The type of the agent, which decides how it runs."
    )
    parser_path: str | None = Field(
        default=None,
        description=(
            "The parser of a TextGuiAgent's replies, as 'module:attribute'; null "
            "removes it. Rejected for every other type."
        ),
    )


class IgnoredField(SQLModel):
    """A field an agent stores but its type doesn't use."""

    field: str = Field(description="The field's name.")
    reason: str = Field(description="Why the agent's type doesn't use it.")


def _saved_agent_model() -> ModelClass:
    """An agent as create and update return it: its columns and ``ignored_fields``."""
    fields: dict[str, Any] = {
        name: (info.annotation, info) for name, info in Agent.model_fields.items()
    }
    fields["ignored_fields"] = (
        list[IgnoredField],
        Field(
            default_factory=list,
            description="The fields the agent's type stores but doesn't use.",
        ),
    )
    return create_model(
        "SavedAgent",
        __base__=SQLModel,
        __doc__="An agent as saved, with the fields its type doesn't use.",
        **fields,
    )


SavedAgent = _saved_agent_model()
"""The response of agent create and update; a table model can't be subclassed."""


class SubAgent(SQLModel, table=True):
    # In the database, not in __init__: model_validate() builds an empty instance
    # first, on which a rule across fields would run.
    __table_args__ = (
        CheckConstraint("parent_agent_id <> child_agent_id", name="subagent_not_self"),
    )

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the sub-agent.",
        primary_key=True,
    )
    parent_agent_id: uuid.UUID = Field(
        foreign_key="agent.id",
        description="Foreign key to the parent agent.",
    )
    parent_agent: Agent = Relationship(
        back_populates="sub_agents",
        sa_relationship_kwargs={
            "lazy": "joined",
            "foreign_keys": "SubAgent.parent_agent_id",
        },
    )
    child_agent_id: uuid.UUID = Field(
        foreign_key="agent.id",
        description="Foreign key to the child agent.",
    )
    child_agent: Agent = Relationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "foreign_keys": "SubAgent.child_agent_id",
        },
    )

    limit: int | None = Field(
        default=None,
        description="Optional limit on the number of times the sub-agent can be called by the parent agent.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the sub-agent association was created.",
    )


class AgentTool(SQLModel, table=True):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for the agent-tool association.",
        primary_key=True,
    )
    agent_id: uuid.UUID = Field(
        foreign_key="agent.id",
        description="Foreign key to the agent.",
    )
    agent: Agent = Relationship(
        back_populates="tools",
        sa_relationship_kwargs={"lazy": "joined"},
    )
    tool_id: uuid.UUID = Field(
        foreign_key="tool.id",
        description="Foreign key to the tool.",
    )
    tool: Tool = Relationship(
        sa_relationship_kwargs={"lazy": "joined"},
    )

    limit: int | None = Field(
        None,
        description="Optional limit on the number of times the tool can be called by the agent.",
    )
    required: bool = Field(
        False,
        description="Indicates whether the tool must be called by the agent at least once.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the agent-tool association was created.",
    )


def _input_errors(exc: ValidationError) -> list[str]:
    """One message per invalid input, naming the argument."""
    messages: list[str] = []
    for error in exc.errors():
        name, *inner = (str(part) for part in error["loc"])
        match error["type"]:
            case "missing" if not inner:
                messages.append(f"Missing required argument: {name}")
            case "extra_forbidden" if not inner:
                messages.append(f"Unexpected argument: {name}")
            case _:
                at = f" at {'.'.join(inner)}" if inner else ""
                messages.append(f"Argument '{name}'{at}: {error['msg']}")
    return messages
