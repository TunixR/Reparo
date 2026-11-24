"""A recovery runs the gateway's enabled agent graph from the database.

The node agents are real ``Agent`` rows run by real Strands agents over scripted
models, so input binding, ``set_continuation`` and the shared invocation state run
as they do in production.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from database.agent_graphs.runner import AGENT_GRAPH_CONTEXT_KEY, NodeRun
from database.agents.behaviours import AgentBehaviour, DefaultAgent
from database.agents.models import Agent, AgentTool, AgentType
from database.logging.models import RobotException
from database.provider.models import Router
from database.tools.models import Tool
from tests.unit.agent_tools._shared.process_models import SPENDING_APPROVAL_MODEL
from tests.unit.fixtures.agent_graph_fixtures import (
    Edge,
    MakeAgentGraph,
    Node,
    argument,
    output_class,
    output_model,
)
from tests.unit.shared.log_events import logged_events
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn

GATEWAY = "Gateway Orchestrator"
INFERER = "Recovery Inferer"
CONTINUATION = "Continuation Inferer"


TRIAGE = output_model("Triage", diagnosis="str", ui_exception="bool")
INFERENCE = output_model("Inference", recovery_task="str")
REPORT = output_model(
    "UiExceptionReport",
    result="str",
    success="bool",
    process_finished="bool",
    continue_from_model_event_id="str | None",
    loop_iteration="list[int]",
)
Triage = output_class(TRIAGE)
Inference = output_class(INFERENCE)


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-graph-0001"
    _ = make_robot_key(name="Graph", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def payload(valid_recovery_payload: dict[str, Any]) -> dict[str, Any]:
    return valid_recovery_payload | {"model": SPENDING_APPROVAL_MODEL}


@pytest.fixture
def agents(session: Session, mock_router: Router) -> dict[str, Agent]:
    """A gateway, an inferring error agent and a continuation-setting agent."""

    def _agent(
        name: str, type_: AgentType, model: dict[str, Any], *arguments: str
    ) -> Agent:
        return Agent(
            id=uuid.uuid4(),
            name=name,
            description=name,
            prompt=f"You are the {name}.",
            output_model=model,
            input_type=Agent.InputType.TEXT,
            router=mock_router,
            router_id=mock_router.id,
            type=type_,
            arguments=[argument(name) for name in arguments],
        )

    gateway = _agent(GATEWAY, AgentType.Agent, TRIAGE, "task_name")
    inferer = _agent(INFERER, AgentType.Agent, INFERENCE, "task_name", "symptom")
    continuation = _agent(
        CONTINUATION,
        AgentType.Agent,
        REPORT,
        "recovery_task",
    )
    tool = Tool(
        id=uuid.uuid4(),
        name="set_continuation",
        description="Set where the robot continues.",
        fn_module="agent_tools.utils.set_continuation",
    )
    session.add_all([gateway, inferer, continuation, tool])
    session.commit()
    session.add(
        AgentTool(agent_id=continuation.id, tool_id=tool.id, limit=None, required=True)
    )
    session.commit()
    for agent in (gateway, inferer, continuation):
        session.refresh(agent)
    return {agent.name: agent for agent in (gateway, inferer, continuation)}


@pytest.fixture
def script(
    agents: dict[str, Agent], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, ScriptedModel]]:
    """Give each agent, by name, a scripted model."""

    def _script(**turns: list[Turn]) -> dict[str, ScriptedModel]:
        models = {
            name: ScriptedModel(*turns.get(key, ["Unused."]))
            for key, name in (
                ("gateway", GATEWAY),
                ("inferer", INFERER),
                ("continuation", CONTINUATION),
            )
        }
        monkeypatch.setattr(Agent, "get_model", lambda self: models[self.name])  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        monkeypatch.setattr(DefaultAgent, "logging_hooks", lambda self, state: [])  # pyright: ignore[reportUnknownLambdaType]
        monkeypatch.setattr(Router, "get_conversation_cost", lambda self, i, o: 0.0)  # pyright: ignore[reportUnknownLambdaType]
        return models

    return _script


@pytest.fixture
def node_runs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Each node run, in order: its agent's name and the state it saw."""
    runs: list[tuple[str, dict[str, Any]]] = []
    original = AgentBehaviour.run_node

    async def _spy(
        self: AgentBehaviour,
        invocation_state: dict[str, Any],
        input_map: Mapping[str, str],
    ) -> NodeRun:
        runs.append((self.agent.name, invocation_state))
        return await original(self, invocation_state, input_map)

    monkeypatch.setattr(AgentBehaviour, "run_node", _spy)
    return runs


def _recover(
    client: TestClient, robot_key_raw: str, payload: dict[str, Any]
) -> dict[str, Any]:
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": robot_key_raw}
    ) as ws:
        ws.send_json(payload)
        while True:
            message = ws.receive_json()
            if message["type"] != "ping":
                return message


def _instruction(model: ScriptedModel, run: int = 0) -> str:
    """The ``name: value`` lines an agent run got, the first user text after the prompt."""
    texts = [
        block["text"]
        for message in model.requests[run]
        if message["role"] == "user"
        for block in message["content"]
        if "text" in block
    ]
    return texts[1]


def _triage(
    diagnosis: str = "Submit is disabled", *, ui_exception: bool = True
) -> list[Turn]:
    return [
        "Triaged.",
        ToolCall("Triage", {"diagnosis": diagnosis, "ui_exception": ui_exception}),
    ]


def _inference(recovery_task: str = "Enable and click Submit") -> list[Turn]:
    return ["Inferred.", ToolCall("Inference", {"recovery_task": recovery_task})]


def _continued(model_event_id: str = "click_approve") -> list[Turn]:
    return [
        ToolCall(
            "set_continuation",
            {
                "recovered": True,
                "process_finished": False,
                "model_event_id": model_event_id,
                "loop_iteration": [2],
            },
        ),
        "Continuation set.",
        ToolCall(
            "UiExceptionReport",
            {
                "result": "done",
                "success": True,
                "process_finished": False,
                "continue_from_model_event_id": None,
                "loop_iteration": [],
            },
        ),
    ]


@pytest.fixture
def linear_graph(
    agents: dict[str, Agent], make_agent_graph: MakeAgentGraph
) -> Callable[..., None]:
    """Persist gateway -> inferer -> continuation, with the given graph settings."""

    def _linear_graph(**settings: Any) -> None:
        _ = make_agent_graph(
            Node("gateway", agents[GATEWAY]),
            Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
            Node("continuation", agents[CONTINUATION]),
            **settings,
        )

    return _linear_graph


def test_a_linear_graph_runs_every_node_and_answers_from_the_last(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    linear_graph: Callable[..., None],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    linear_graph()
    models = script(gateway=_triage(), inferer=_inference(), continuation=_continued())

    message = _recover(client, robot_key_raw, payload)

    assert [name for name, _ in node_runs] == [GATEWAY, INFERER, CONTINUATION]
    assert message["type"] == "done"
    assert message["content"] == {
        "success": True,
        "process_finished": False,
        "continue_from_model_event_id": "click_approve",
        "loop_iteration": [2],
        "variables": payload["variables"],
    }
    assert _instruction(models[GATEWAY]) == "task_name: Invoice Processing"
    assert _instruction(models[INFERER]) == (
        "task_name: Invoice Processing\nsymptom: Submit is disabled"
    )
    assert _instruction(models[CONTINUATION]) == (
        "recovery_task: Enable and click Submit"
    )


def test_one_agent_backing_two_nodes_runs_once_per_node(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("infer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        Node("check", agents[INFERER], {"symptom": "infer.recovery_task"}),
    )
    models = script(
        gateway=_triage(),
        inferer=[*_inference("First guess"), *_inference("Second guess")],
    )

    message = _recover(client, robot_key_raw, payload)

    assert message["type"] == "done"
    assert message["content"]["success"] is False
    assert [name for name, _ in node_runs] == [GATEWAY, INFERER, INFERER]
    assert _instruction(models[INFERER], run=0).endswith("symptom: Submit is disabled")
    assert _instruction(models[INFERER], run=2).endswith("symptom: First guess")
    context = node_runs[-1][1][AGENT_GRAPH_CONTEXT_KEY]
    assert context.outputs == {
        "gateway": Triage.model_validate(
            {"diagnosis": "Submit is disabled", "ui_exception": True}
        ),
        "infer": Inference.model_validate({"recovery_task": "First guess"}),
        "check": Inference.model_validate({"recovery_task": "Second guess"}),
    }


def test_leftover_state_is_cleaned_but_never_copied_between_nodes(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    linear_graph: Callable[..., None],
    node_runs: list[tuple[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
):
    linear_graph()
    _ = script(gateway=_triage(), inferer=_inference(), continuation=_continued())
    seen: list[set[str]] = []
    spy = AgentBehaviour.run_node

    async def _leave_leftovers(
        self: AgentBehaviour,
        invocation_state: dict[str, Any],
        input_map: Mapping[str, str],
    ) -> NodeRun:
        seen.append(set(invocation_state))
        run = await spy(self, invocation_state, input_map)
        invocation_state.update(
            {"parent_trace_id": "x", "event_loop_cycle_id": "y", "agent": "z"}
        )
        return run

    monkeypatch.setattr(AgentBehaviour, "run_node", _leave_leftovers)

    message = _recover(client, robot_key_raw, payload)

    assert message["content"]["success"] is True
    leftovers = {"parent_trace_id", "event_loop_cycle_id", "agent", "request_state"}
    assert all(not keys & leftovers for keys in seen)
    assert len({id(state) for _, state in node_runs}) == 1


def test_hitting_max_node_executions_is_done_without_success(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    linear_graph: Callable[..., None],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    linear_graph(max_node_executions=2)
    _ = script(gateway=_triage(), inferer=_inference(), continuation=_continued())

    message = _recover(client, robot_key_raw, payload)

    assert message["type"] == "done"
    assert message["content"]["success"] is False
    assert [name for name, _ in node_runs] == [GATEWAY, INFERER]


def test_a_passed_execution_timeout_is_done_without_success(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    linear_graph: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
):
    linear_graph(execution_timeout_s=0.01)
    ran: list[str] = []

    async def _slow_node(
        self: AgentBehaviour, _state: dict[str, Any], _input_map: Mapping[str, str]
    ) -> NodeRun:
        ran.append(self.agent.name)
        await asyncio.sleep(0.05)
        return NodeRun(
            output=Triage.model_validate({"diagnosis": "slow", "ui_exception": True})
        )

    monkeypatch.setattr(AgentBehaviour, "run_node", _slow_node)

    message = _recover(client, robot_key_raw, payload)

    assert message["type"] == "done"
    assert message["content"]["success"] is False
    assert ran == [GATEWAY]


def test_a_node_timeout_is_an_internal_error(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    linear_graph: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
):
    linear_graph(node_timeout_s=0.05)

    async def _hanging_node(
        _self: AgentBehaviour, _state: dict[str, Any], _input_map: Mapping[str, str]
    ) -> NodeRun:
        await asyncio.sleep(1)
        return NodeRun(
            output=Triage.model_validate(
                {"diagnosis": "too late", "ui_exception": True}
            )
        )

    monkeypatch.setattr(AgentBehaviour, "run_node", _hanging_node)

    message = _recover(client, robot_key_raw, payload)

    assert message == {
        "type": "error",
        "code": "INTERNAL_ERROR",
        "content": "Internal server error",
    }


def test_an_exception_in_a_later_node_is_an_internal_error(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    linear_graph: Callable[..., None],
):
    linear_graph()
    models = script(gateway=_triage(), inferer=[RuntimeError("model down")])

    message = _recover(client, robot_key_raw, payload)

    assert message["code"] == "INTERNAL_ERROR"
    assert models[CONTINUATION].calls == 0


@pytest.mark.parametrize("disabled", [INFERER, CONTINUATION])
def test_the_next_node_is_cancelled_once_a_graph_agent_is_disabled_mid_run(
    disabled: str,
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    session: Session,
    agents: dict[str, Agent],
    script: Callable[..., dict[str, ScriptedModel]],
    linear_graph: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    linear_graph()
    models = script(gateway=_triage(), inferer=_inference(), continuation=_continued())
    original = AgentBehaviour.run_node

    async def _disable_an_agent_after_the_gateway(
        self: AgentBehaviour,
        invocation_state: dict[str, Any],
        input_map: Mapping[str, str],
    ) -> NodeRun:
        run = await original(self, invocation_state, input_map)
        if self.agent.name == GATEWAY:
            agent = agents[disabled]
            agent.enabled = False
            session.add(agent)
            session.commit()
        return run

    monkeypatch.setattr(AgentBehaviour, "run_node", _disable_an_agent_after_the_gateway)

    message = _recover(client, robot_key_raw, payload)

    assert message["type"] == "error"
    assert message["code"] == "INTERNAL_ERROR"
    assert models[GATEWAY].calls > 0
    assert models[INFERER].calls == 0
    [cancelled] = logged_events(caplog, "agent_graph.node_cancelled")
    assert vars(cancelled)["agent_graph.node.key"] == "inferer"
    assert vars(cancelled)["agent.id"] == str(agents[INFERER].id)


def test_the_run_starts_at_the_entry_node_even_inside_a_cycle(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        edges=[("gateway", "inferer"), ("inferer", "gateway")],
        max_node_executions=3,
    )
    _ = script(gateway=[*_triage(), *_triage()], inferer=_inference())

    message = _recover(client, robot_key_raw, payload)

    assert message["type"] == "done"
    assert message["content"]["success"] is False
    assert [name for name, _ in node_runs] == [GATEWAY, INFERER, GATEWAY]


UI_EXCEPTION = {"field": "ui_exception", "op": "eq", "value": True}


@pytest.fixture
def ui_exception_graph(
    agents: dict[str, Agent], make_agent_graph: MakeAgentGraph
) -> None:
    """Persist gateway -> inferer -> continuation, past the gateway only on UI exceptions."""
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        Node("continuation", agents[CONTINUATION]),
        edges=[
            Edge("gateway", "inferer", UI_EXCEPTION),
            Edge("inferer", "continuation"),
        ],
    )


@pytest.mark.usefixtures("ui_exception_graph")
def test_a_satisfied_condition_runs_the_recovery_path(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    _ = script(
        gateway=_triage(ui_exception=True),
        inferer=_inference(),
        continuation=_continued(),
    )

    message = _recover(client, robot_key_raw, payload)

    assert [name for name, _ in node_runs] == [GATEWAY, INFERER, CONTINUATION]
    assert message["type"] == "done"
    assert message["content"]["success"] is True


@pytest.mark.usefixtures("ui_exception_graph")
def test_an_unsatisfied_condition_ends_after_the_gateway_without_success(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    _ = script(gateway=_triage(ui_exception=False))

    message = _recover(client, robot_key_raw, payload)

    assert [name for name, _ in node_runs] == [GATEWAY]
    assert message["type"] == "done"
    assert message["content"]["success"] is False


def _condition_errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return logged_events(caplog, "agent_graph.condition_error")


def test_of_two_satisfied_edges_only_the_higher_priority_target_runs(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
):
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        Node(
            "continuation", agents[CONTINUATION], {"recovery_task": "gateway.diagnosis"}
        ),
        edges=[
            Edge("gateway", "inferer", UI_EXCEPTION, priority=2),
            Edge("gateway", "continuation", UI_EXCEPTION, priority=1),
            Edge(
                "gateway",
                "gateway",
                {"field": "missing", "op": "is_null"},
                priority=3,
            ),
        ],
    )
    _ = script(gateway=_triage(ui_exception=True), continuation=_continued())
    caplog.set_level(logging.WARNING)

    message = _recover(client, robot_key_raw, payload)

    assert [name for name, _ in node_runs] == [GATEWAY, CONTINUATION]
    assert message["content"]["success"] is True
    assert _condition_errors(caplog) == []


def test_the_unconditional_edge_runs_when_nothing_else_matches(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        Node(
            "continuation", agents[CONTINUATION], {"recovery_task": "gateway.diagnosis"}
        ),
        edges=[
            Edge("gateway", "continuation", priority=1),
            Edge("gateway", "inferer", UI_EXCEPTION, priority=0),
        ],
    )
    _ = script(gateway=_triage(ui_exception=False), continuation=_continued())

    message = _recover(client, robot_key_raw, payload)

    assert [name for name, _ in node_runs] == [GATEWAY, CONTINUATION]
    assert message["content"]["success"] is True


def test_a_condition_on_a_missing_field_ends_the_graph_and_is_logged(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
):
    graph = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        edges=[
            Edge("gateway", "inferer", {"field": "severity", "op": "gt", "value": 2})
        ],
    )
    _ = script(gateway=_triage())
    caplog.set_level(logging.WARNING)

    message = _recover(client, robot_key_raw, payload)

    assert [name for name, _ in node_runs] == [GATEWAY]
    assert message["type"] == "done"
    assert message["content"]["success"] is False
    [error] = _condition_errors(caplog)
    assert vars(error)["agent_graph.id"] == str(graph.id)
    assert vars(error)["error.type"] == "KeyError"


@pytest.mark.parametrize("graph", ["disabled_default", "enabled_not_default"])
def test_without_an_enabled_default_recovery_graph_nothing_runs(
    graph: str,
    session: Session,
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    node_runs: list[tuple[str, dict[str, Any]]],
):
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        enabled=graph == "enabled_not_default",
        is_default_recovery=graph == "disabled_default",
    )
    _ = script(gateway=_triage())

    message = _recover(client, robot_key_raw, payload)

    assert message == {
        "type": "error",
        "code": "NO_AGENT_GRAPH",
        "content": "No enabled default recovery agent graph in the database.",
    }
    assert node_runs == []
    assert session.exec(select(RobotException)).first() is None
