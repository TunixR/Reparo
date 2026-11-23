import enum
import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, override

from pydantic import BaseModel, TypeAdapter, ValidationError
from sqlalchemy import Column, Dialect, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Enum, Field, Relationship, SQLModel
from strands import Agent as StrandsAgent
from strands import ToolContext, tool
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.hooks import HookProvider
from strands.models import Model
from strands.tools.decorator import DecoratedFunctionTool
from strands.types.content import Messages

from agent_tools.hooks import (
    AgentLoggingHook,
    LimitToolCounts,
    RequireToolCalls,
    ToolLoggingHook,
)
from agent_tools.image import screenshot_bytes
from database.agent_graphs.runner import (
    AGENT_GRAPH_CONTEXT_KEY,
    AgentGraphContext,
    NodeRun,
)
from database.agents.specs import (
    FIELD_SPECS,
    FieldSpec,
    FieldSpecs,
    OutputModelSpec,
    build_input_model,
    build_output_model,
)
from database.agents.tooled_gui_agent import TOOL_NAME as GUI_AGENT_TOOL_NAME
from database.agents.tooled_gui_agent import GuiRunReport, TooledGuiAgent
from database.provider.models import Router
from database.tools.models import Tool
from observability import attributes as attrs
from observability.events import log_event

logger = logging.getLogger(__name__)

# Module level: inside `Agent` the name `type` is the agent type field.
type ModelClass = type[BaseModel]

_UNBOUND = object()
"""What the graph context gives for an argument it has no value for."""


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
    Agent = "Agent"
    GuiAgent = "GuiAgent"


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

    traces: list["AgentTrace"] = Relationship(  # noqa: F821 # pyright:ignore[reportUndefinedVariable]
        back_populates="agent",
        sa_relationship_kwargs={"lazy": "joined"},
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the agent was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the agent was last updated.",
    )

    type: AgentType = Field(  # Until polymorphism is supported by sqlmodel
        ...,
        sa_column=Column(
            Enum(AgentType),
            nullable=False,
            default=AgentType.Agent,
        ),
        description="The type of the agent.",
    )

    # __mapper_args__ = {
    #     "polymorphic_identity": "Agent",
    #     "polymorphic_abstract": True,
    #     "polymorphic_on": "type",
    # }

    def get_tool_name(self) -> str:
        """Return the name of the tool this agent is exposed as (see `as_tool`)."""
        if self.type == AgentType.GuiAgent:
            return GUI_AGENT_TOOL_NAME
        return self.name.lower().replace(" ", "_")

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

    def get_tool_limiter(self) -> LimitToolCounts:
        """Return a tool limiter hook provider based on the agent's tool limits."""
        tool_limits = {}
        for at in self.tools:
            if at.limit is not None:
                tool_limits[at.tool.name] = at.limit
        for sa in self.sub_agents:
            if sa.limit is not None:
                tool_limits[sa.child_agent.get_tool_name()] = sa.limit
        return LimitToolCounts(tool_limits)

    def get_required_tools_hook(self) -> RequireToolCalls:
        """Return a hook that sends the agent back until its required tools are called."""
        return RequireToolCalls([at.tool.name for at in self.tools if at.required])

    def get_logging_hooks(self, invocation_state: dict[str, Any]) -> list[HookProvider]:
        agent_logging_hook = AgentLoggingHook(
            agent_id=self.id,
            invocation_state=invocation_state,
            parent_trace_id=invocation_state.get("parent_trace_id", None),
            is_gui_agent=self.type == AgentType.GuiAgent,
        )
        invocation_state["parent_trace_id"] = agent_logging_hook.agent_trace_id

        tools = {at.tool.name: at.tool.id for at in self.tools}
        tool_logging_hook = ToolLoggingHook(
            agent_trace_id=agent_logging_hook.agent_trace_id,
            tools=tools,
        )

        return [agent_logging_hook, tool_logging_hook]

    def as_tool(self) -> DecoratedFunctionTool:  # pyright: ignore[reportMissingTypeArgument]
        """Dynamically creates the agent tool function for usage within other agents."""

        ## Temporary fix until polymorphism is supported by sqlmodel
        if self.type == AgentType.GuiAgent:  # Actually only supports uitars
            return TooledGuiAgent(self).tooled_guiagent

        @tool(
            name=self.get_tool_name(),
            description=self.description,
            inputSchema=self.get_input_schema(),
            context=True,
        )
        async def agent_tool_function(
            tool_context: ToolContext,
        ) -> str | dict[str, Any]:
            if "websocket" not in tool_context.invocation_state:
                raise ValueError(
                    "WebSocket must be provided in tool context invocation state."
                )
            if "robot_exception_id" not in tool_context.invocation_state:
                raise ValueError(
                    "A valid robot_exception_id must be provided in tool context invocation state."
                )

            return await self(
                invocation_state=tool_context.invocation_state,
                **tool_context.tool_use.get("input", {}),
            )

        return agent_tool_function

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
        """Apply an update payload, validating the resulting router alignment."""
        data = {
            key: value
            for key, value in payload.model_dump(exclude_unset=True).items()
            if value is not None
        }
        input_type = data.get("input_type", self.input_type)
        self.validate_capabilities_support_input_type(
            router.capabilities,
            input_type,
            router_name=router.model_name,
        )

        # An explicit null clears the effort back to the provider default, and the
        # output model to none.
        for key in ("reasoning_effort", "output_model"):
            if key in payload.model_fields_set:
                data[key] = getattr(payload, key)

        for key, value in data.items():
            try:
                setattr(self, key, value)
            except Exception as e:
                raise ValueError(f"Failed to set field '{key}': {e!s}") from e

        self.router = router
        self.router_id = router.id

    def output_class(self) -> ModelClass | None:
        """Return the class of this agent's structured output, None if it has none.

        A GUI agent reports a `GuiRunReport`; any other agent the class built from its
        ``output_model``.
        """
        if self.type == AgentType.GuiAgent:
            return GuiRunReport
        if self.output_model is None:
            return None
        return build_output_model(self.output_model)

    async def __call__(
        self, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Run the agent on explicit inputs and return its structured response as a dict.

        This is how an agent runs standalone and as a tool of another agent.
        """
        if invocation_state is None:
            invocation_state = {}
        return (await self._execute(invocation_state, kwargs)).model_dump()

    async def run_node(
        self, invocation_state: dict[str, Any], input_map: Mapping[str, str]
    ) -> NodeRun:
        """Run the agent as a node of an agent graph and return its structured output.

        Each argument comes from the graph context at its ``input_map`` path, else by
        its name. Like ``__call__``, it records the inputs in ``invocation_state``. A
        GUI agent drives the robot's desktop and reports a `GuiRunReport`.
        """
        context: AgentGraphContext = invocation_state[AGENT_GRAPH_CONTEXT_KEY]
        # An argument nothing in the context has stays unbound, so validation reports
        # it as missing; a None found in the context is a value.
        inputs = {
            a.name: value
            for a in self.arguments
            if (value := context.get(input_map.get(a.name, a.name), _UNBOUND))
            is not _UNBOUND
        }
        if self.type == AgentType.GuiAgent:
            bound = self._bind_inputs(invocation_state, inputs)
            report = await TooledGuiAgent(self).run(invocation_state, bound)
            return NodeRun(output=report)
        return NodeRun(output=await self._execute(invocation_state, inputs))

    def _bind_inputs(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate ``inputs``, record them in ``invocation_state`` for the agent trace,
        and return them."""
        bound = self.validate_input(**inputs)
        invocation_state["inputs"] = bound
        return bound

    async def _execute(
        self, invocation_state: dict[str, Any], inputs: Mapping[str, Any]
    ) -> BaseModel:
        """Run the agent on validated inputs and return its structured output.

        Tools, sub-agents, hooks and the output model are computed per call. The agent
        runs twice on one conversation: a free run on the inputs, then a
        structured-output call. Usage and cost of both are logged, even on failure.
        """
        inputs = self._bind_inputs(invocation_state, inputs)

        tools: list[DecoratedFunctionTool] = [  # pyright: ignore[reportMissingTypeArgument]
            t.get_tool_function() for t in self.get_tools()
        ]
        sub_agent_tools: list[DecoratedFunctionTool] = [  # pyright: ignore[reportMissingTypeArgument]
            sa.as_tool() for sa in self.get_sub_agents()
        ]

        messages: Messages = [
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {
                                "bytes": await screenshot_bytes(
                                    invocation_state["websocket"]
                                ),
                            },
                        },
                    },
                ]
                if self.input_type == self.InputType.IMAGETEXT
                else [
                    {
                        "text": self.prompt,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": "I understand the instructions. I will proceed once you give me all neccesary values.",
                    }
                ],
            },
        ]

        model = self.get_model()

        hooks = [
            self.get_tool_limiter(),
            self.get_required_tools_hook(),
            *self.get_logging_hooks(invocation_state),
        ]

        strands_agent = StrandsAgent(
            model=model,
            tools=tools + sub_agent_tools,
            hooks=hooks,
            system_prompt=self.prompt,
            messages=messages,  # type: ignore This works fine
            # Strands' default window, with the seeded pair and the instruction
            # pinned: it trims when the first run ends, and the structured-output
            # call still has to see the inputs.
            conversation_manager=SlidingWindowConversationManager(
                pin_first=len(messages) + 1
            ),
        )

        try:
            instruction = "\n".join(
                [f"{arg.name}: {inputs.get(arg.name)}" for arg in self.arguments]
            )

            _ = await strands_agent.invoke_async(
                instruction,
                invocation_state=invocation_state,
            )

            response = (
                await strands_agent.invoke_async(
                    "Given our conversation so far, please provide a report using structured output with the given tool.",
                    structured_output_model=self.output_class(),
                    invocation_state=invocation_state,
                )
            ).structured_output

            if response is None:
                raise ValueError("Agent did not return a structured response.")
            return response
        finally:
            # Every AgentResult.metrics is this same object; accumulated_usage sums
            # both invocations above, including tokens spent before a failure.
            usage = strands_agent.event_loop_metrics.accumulated_usage
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)
            cost = self.router.get_conversation_cost(input_tokens, output_tokens)
            agent_logging_hook: AgentLoggingHook | None = next(
                filter(  # pyright: ignore[reportAssignmentType]
                    lambda h: isinstance(h, AgentLoggingHook), hooks
                ),
                None,
            )
            if agent_logging_hook:
                agent_logging_hook.update_trace(finished=True, cost=cost)
            log_event(
                logger,
                attrs.EVENT_AGENT_USAGE,
                {
                    attrs.AGENT_ID: self.id,
                    attrs.GEN_AI_USAGE_INPUT_TOKENS: input_tokens,
                    attrs.GEN_AI_USAGE_OUTPUT_TOKENS: output_tokens,
                    attrs.AGENT_COST: cost,
                },
            )

    def __init__(self, **data: Any) -> None:
        # Routers are passed as objects so their capabilities can be validated
        # before the agent is attached to a session; a router_id alone would
        # require a database lookup this model cannot perform during __init__.
        router = data.get("router")
        super().__init__(**data)
        if router is not None:
            self.validate_capabilities_support_input_type(
                router.capabilities,
                self.input_type,
                router_name=router.model_name,
            )


class AgentCreate(SQLModel):
    """Schema for creating a new agent."""

    name: str = Field(description="Name of the agent.")
    description: str = Field(description="Description of the agent.")
    prompt: str = Field(description="The prompt used to initialize the agent.")
    arguments: FieldSpecs = Field(
        default_factory=list,
        description="The agent's inputs, in the order its instruction lists them.",
    )
    output_model: OutputModelSpec | None = Field(
        default=None,
        description="The agent's structured output; a GUI agent ignores it.",
    )
    input_type: Agent.InputType = Field(
        default=Agent.InputType.TEXT,
        description="The type of input the agent can process.",
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
        description="The type of the agent.",
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
        description="The agent's inputs, in order; replaces the current list.",
    )
    output_model: OutputModelSpec | None = Field(
        default=None,
        description="The agent's structured output; null removes it.",
    )
    input_type: Agent.InputType | None = Field(
        default=None,
        description="The type of input the agent can process.",
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
    type: AgentType | None = Field(default=None, description="The type of the agent.")


class SubAgent(SQLModel, table=True):
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
        None,
        description="Optional limit on the number of times the sub-agent can be called by the parent agent.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of when the sub-agent association was created.",
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.parent_agent_id == self.child_agent_id:
            raise ValueError("An agent cannot be a sub-agent of itself.")


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
