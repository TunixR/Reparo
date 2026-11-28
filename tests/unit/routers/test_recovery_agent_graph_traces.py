"""A recovery's agent graph run is traced, and the trace is served by the logging API.

The node agents are real ``Agent`` rows run by real Strands agents over scripted
models, with their logging hooks, so the traces are written as in production.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Headers
from sqlmodel import Session

from database.agent_graphs.runner import NodeRun
from database.agents.behaviours import AgentBehaviour
from database.agents.models import Agent, AgentTool, AgentType, SubAgent
from database.logging.models import AgentGraphTrace
from database.provider.models import Router
from database.tools.models import Tool
from tests.unit.agent_tools._shared.process_models import SPENDING_APPROVAL_MODEL
from tests.unit.fixtures.agent_graph_fixtures import (
    Edge,
    MakeAgentGraph,
    Node,
    argument,
    output_model,
)
from tests.unit.shared.pricing import free_of_charge
from tests.unit.shared.scripted_model import ScriptedModel, ToolCall, Turn

GATEWAY = "Gateway Orchestrator"
INFERER = "Recovery Inferer"
HELPER = "Symptom Helper"
CONTINUATION = "Continuation Inferer"

UI_EXCEPTION = {"field": "ui_exception", "op": "eq", "value": True}


TRIAGE = output_model("Triage", diagnosis="str", ui_exception="bool")
INFERENCE = output_model("Inference", recovery_task="str")
HINT = output_model("Hint", hint="str")
REPORT = output_model(
    "UiExceptionReport",
    result="str",
    success="bool",
    process_finished="bool",
    continue_from_model_event_id="str | None",
    loop_iteration="list[int]",
)


@pytest.fixture
def robot_key_raw(make_robot_key) -> str:  # pyright: ignore[reportMissingParameterType]
    key_raw = "test-robot-key-graph-trace-0001"
    _ = make_robot_key(name="Graph traces", enabled=True, key_raw=key_raw)
    return key_raw


@pytest.fixture
def payload(valid_recovery_payload: dict[str, Any]) -> dict[str, Any]:
    return valid_recovery_payload | {"model": SPENDING_APPROVAL_MODEL}


@pytest.fixture
def agents(session: Session, mock_router: Router) -> dict[str, Agent]:
    """A gateway, an inferer with a helper sub-agent, and a continuation setter."""

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
    helper = _agent(HELPER, AgentType.Agent, HINT, "symptom")
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
    session.add_all([gateway, inferer, helper, continuation, tool])
    session.commit()
    session.add_all(
        [
            AgentTool(
                agent_id=continuation.id, tool_id=tool.id, limit=None, required=True
            ),
            SubAgent(parent_agent_id=inferer.id, child_agent_id=helper.id),
        ]
    )
    session.commit()
    everyone = (gateway, inferer, helper, continuation)
    for agent in everyone:
        session.refresh(agent)
    return {agent.name: agent for agent in everyone}


@pytest.fixture
def script(
    agents: dict[str, Agent], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, ScriptedModel]]:
    """Give each agent, by name, a scripted model; logging hooks stay on."""

    def _script(**turns: list[Turn]) -> dict[str, ScriptedModel]:
        models = {
            name: ScriptedModel(*turns.get(key, ["Unused."]))
            for key, name in (
                ("gateway", GATEWAY),
                ("inferer", INFERER),
                ("helper", HELPER),
                ("continuation", CONTINUATION),
            )
        }
        monkeypatch.setattr(Agent, "get_model", lambda self: models[self.name])  # pyright: ignore[reportUnknownLambdaType,reportUnknownMemberType]
        monkeypatch.setattr(Router, "get_conversation_cost", free_of_charge)
        return models

    return _script


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


def _triage(*, ui_exception: bool = True) -> list[Turn]:
    return [
        "Triaged.",
        ToolCall(
            "Triage", {"diagnosis": "Submit is disabled", "ui_exception": ui_exception}
        ),
    ]


def _inference_with_help() -> list[Turn]:
    return [
        ToolCall("symptom_helper", {"symptom": "Submit is disabled"}),
        "Inferred.",
        ToolCall("Inference", {"recovery_task": "Enable and click Submit"}),
    ]


def _hint() -> list[Turn]:
    return ["Hinted.", ToolCall("Hint", {"hint": "Fill the amount first"})]


def _continued() -> list[Turn]:
    return [
        ToolCall(
            "set_continuation",
            {
                "recovered": True,
                "process_finished": False,
                "model_event_id": "click_approve",
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


def _graph_traces(
    client: TestClient, headers: Headers, recovery_id: str | None = None
) -> list[dict[str, Any]]:
    response = client.get(
        "/logging/agent_graph_traces",
        params={} if recovery_id is None else {"robot_exception_id": recovery_id},
        headers=headers,
    )
    assert response.status_code == 200
    return response.json()


def _agent_trace_markdown(
    client: TestClient, headers: Headers, agent_trace_id: str
) -> str:
    response = client.get(
        "/logging/markdown/",
        params={"agent_trace_id": agent_trace_id},
        headers=headers,
    )
    assert response.status_code == 200
    return response.text


@pytest.fixture
def recovery_graph(agents: dict[str, Agent], make_agent_graph: MakeAgentGraph) -> None:
    """Gateway -> inferer (on UI exceptions) -> continuation."""
    _ = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        Node("continuation", agents[CONTINUATION]),
        edges=[
            Edge("gateway", "inferer", UI_EXCEPTION),
            Edge("inferer", "continuation"),
        ],
    )


@pytest.fixture
def inferer_disabled_after_the_gateway(
    session: Session, agents: dict[str, Agent], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disable the inferer once the gateway node has run, so its node is cancelled."""
    original = AgentBehaviour.run_node

    async def _disable_the_inferer_after_the_gateway(
        self: AgentBehaviour,
        invocation_state: dict[str, Any],
        input_map: Mapping[str, str],
    ) -> NodeRun:
        run = await original(self, invocation_state, input_map)
        inferer = agents[INFERER]
        inferer.enabled = False
        session.add(inferer)
        session.commit()
        return run

    monkeypatch.setattr(
        AgentBehaviour, "run_node", _disable_the_inferer_after_the_gateway
    )


@pytest.mark.usefixtures("recovery_graph")
def test_a_run_is_one_agent_graph_trace_with_flat_ordered_node_traces(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    script: Callable[..., dict[str, ScriptedModel]],
    user_headers: Headers,
):
    _ = script(
        gateway=_triage(),
        inferer=_inference_with_help(),
        helper=_hint(),
        continuation=_continued(),
    )

    message = _recover(client, robot_key_raw, payload)

    assert message["type"] == "done"
    [trace] = _graph_traces(client, user_headers, message["id"])
    assert trace["status"] == "completed"
    assert trace["failure_reason"] is None
    assert trace["terminal_node_key"] == "continuation"
    assert trace["answer"] == message["content"]
    assert trace["finished_at"] is not None
    nodes = trace["node_traces"]
    assert [(n["execution_index"], n["node_key"], n["status"]) for n in nodes] == [
        (0, "gateway", "completed"),
        (1, "inferer", "completed"),
        (2, "continuation", "completed"),
    ]
    assert [n["entry_condition"] for n in nodes] == [None, UI_EXCEPTION, None]
    assert nodes[0]["entry_edge_id"] is None
    assert all(n["entry_edge_id"] for n in nodes[1:])

    agent_traces = [
        client.get(
            f"/logging/agent_traces/{n['agent_trace_id']}", headers=user_headers
        ).json()
        for n in nodes
    ]
    assert [t["agent_id"] for t in agent_traces] == [
        str(agents[name].id) for name in (GATEWAY, INFERER, CONTINUATION)
    ]
    markdowns = [
        _agent_trace_markdown(client, user_headers, n["agent_trace_id"]) for n in nodes
    ]
    assert "Sub-Agent Trace" not in markdowns[0]
    assert f"Sub-Agent Trace: {HELPER}" in markdowns[1]
    assert "Sub-Agent Trace" not in markdowns[2]


@pytest.mark.usefixtures("recovery_graph")
def test_a_run_that_raises_leaves_a_failed_trace_and_an_internal_error(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    user_headers: Headers,
):
    _ = script(gateway=_triage(), inferer=[RuntimeError("model down")])

    message = _recover(client, robot_key_raw, payload)

    assert message == {
        "type": "error",
        "code": "INTERNAL_ERROR",
        "content": "Internal server error",
    }
    [trace] = _graph_traces(client, user_headers)
    assert trace["status"] == "failed"
    assert trace["failure_reason"] == "RuntimeError: model down"
    assert trace["answer"] is None
    assert trace["finished_at"] is not None
    gateway, inferer = trace["node_traces"]
    assert gateway["status"] == "completed"
    assert inferer["node_key"] == "inferer"
    assert inferer["status"] == "failed"
    assert inferer["error"] == "RuntimeError: model down"
    assert inferer["agent_trace_id"] is not None


@pytest.mark.usefixtures("recovery_graph", "inferer_disabled_after_the_gateway")
def test_a_node_cancelled_mid_run_is_traced_without_an_agent_trace(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    user_headers: Headers,
):
    _ = script(gateway=_triage())

    message = _recover(client, robot_key_raw, payload)

    assert message["code"] == "INTERNAL_ERROR"
    [trace] = _graph_traces(client, user_headers)
    assert trace["status"] == "failed"
    assert trace["failure_reason"].startswith("NodeCancelledError: ")
    _, inferer = trace["node_traces"]
    assert inferer["status"] == "cancelled"
    assert inferer["agent_trace_id"] is None


def _set_null_as_postgres_does(session: Session, trace_id: str) -> None:
    """Apply the trace's ``ON DELETE SET NULL`` keys, which SQLite doesn't enforce."""
    trace = session.get_one(AgentGraphTrace, uuid.UUID(trace_id))
    session.refresh(trace)
    trace.agent_graph_id = None
    for node_trace in trace.node_traces:
        node_trace.entry_edge_id = None
    session.add(trace)
    session.commit()


def test_the_copied_entry_condition_survives_editing_and_deleting_the_graph(
    session: Session,
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    user_headers: Headers,
    admin_headers: Headers,
):
    graph = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        edges=[Edge("gateway", "inferer", UI_EXCEPTION)],
    )
    _ = script(gateway=_triage(), inferer=_inference_with_help(), helper=_hint())
    message = _recover(client, robot_key_raw, payload)
    [trace] = _graph_traces(client, user_headers, message["id"])
    edge_id = trace["node_traces"][1]["entry_edge_id"]

    disabled = client.patch(
        f"/agent_graphs/{graph.id}", json={"enabled": False}, headers=admin_headers
    )
    assert disabled.status_code == 200
    edited = client.patch(
        f"/agent_graphs/{graph.id}/edges/{edge_id}",
        json={"condition": {"field": "ui_exception", "op": "eq", "value": False}},
        headers=admin_headers,
    )
    assert edited.status_code == 200
    after_edit = client.get(
        f"/logging/agent_graph_traces/{trace['id']}", headers=user_headers
    ).json()
    assert after_edit["node_traces"][1]["entry_condition"] == UI_EXCEPTION

    deleted = client.delete(f"/agent_graphs/{graph.id}", headers=admin_headers)
    assert deleted.status_code == 204
    _set_null_as_postgres_does(session, trace["id"])
    after_delete = client.get(
        f"/logging/agent_graph_traces/{trace['id']}", headers=user_headers
    ).json()
    assert after_delete["node_traces"][1]["entry_condition"] == UI_EXCEPTION
    markdown = _graph_markdown(client, user_headers, trace["id"])
    assert "**Agent Graph:** deleted\n\n" in markdown
    assert (
        "**Entered By:** a deleted edge\n\n"
        '**Condition:** `{"field": "ui_exception", "op": "eq", "value": true}`\n\n'
    ) in markdown
    assert [n["node_key"] for n in after_delete["node_traces"]] == [
        "gateway",
        "inferer",
    ]


def _graph_markdown(client: TestClient, headers: Headers, trace_id: str) -> str:
    response = client.get(
        f"/logging/agent_graph_traces/{trace_id}/markdown", headers=headers
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    return response.text


def _split_nodes(markdown: str) -> list[str]:
    """The header, then one section per node, each starting at its heading."""
    return re.split(r"\n(?=## Node \d+: )", markdown)


def test_the_markdown_export_is_the_header_then_each_node_with_its_agent_trace(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    script: Callable[..., dict[str, ScriptedModel]],
    user_headers: Headers,
):
    graph = make_agent_graph(
        Node("gateway", agents[GATEWAY]),
        Node("inferer", agents[INFERER], {"symptom": "gateway.diagnosis"}),
        Node("continuation", agents[CONTINUATION]),
        edges=[
            Edge("gateway", "inferer", UI_EXCEPTION),
            Edge("inferer", "continuation"),
        ],
        name="UI Recovery",
    )
    _ = script(
        gateway=_triage(),
        inferer=_inference_with_help(),
        helper=_hint(),
        continuation=_continued(),
    )
    message = _recover(client, robot_key_raw, payload)
    [trace] = _graph_traces(client, user_headers, message["id"])
    nodes = trace["node_traces"]

    markdown = _graph_markdown(client, user_headers, trace["id"])

    header, *sections = _split_nodes(markdown)
    assert header.startswith(
        f"# Agent Graph Trace: {trace['id']}\n\n"
        f"**Agent Graph:** UI Recovery ({graph.id})\n\n"
        f"**Robot Exception:** {message['id']}\n\n"
        "**Status:** completed\n\n"
        "**Terminal Node:** continuation\n\n"
    )
    assert header.endswith(
        f"**Answer:**\n```json\n{json.dumps(message['content'], indent=2)}\n```\n"
    )
    entries = [
        ("the entry node", "none"),
        (
            f"edge {nodes[1]['entry_edge_id']}",
            '`{"field": "ui_exception", "op": "eq", "value": true}`',
        ),
        (f"edge {nodes[2]['entry_edge_id']}", "unconditional"),
    ]
    assert sections == [
        f"## Node {index}: {node['node_key']}\n\n"
        f"**Agent:** {name} ({agents[name].id})\n\n"
        f"**Entered By:** {entered_by}\n\n"
        f"**Condition:** {condition}\n\n"
        "**Status:** completed\n\n"
        + _agent_trace_markdown(client, user_headers, node["agent_trace_id"])
        for index, (node, name, (entered_by, condition)) in enumerate(
            zip(nodes, (GATEWAY, INFERER, CONTINUATION), entries, strict=True)
        )
    ]


@pytest.mark.usefixtures("recovery_graph", "inferer_disabled_after_the_gateway")
def test_the_markdown_export_shows_why_a_run_failed_and_steps_without_agent_traces(
    client: TestClient,
    robot_key_raw: str,
    payload: dict[str, Any],
    script: Callable[..., dict[str, ScriptedModel]],
    user_headers: Headers,
):
    _ = script(gateway=_triage())
    _ = _recover(client, robot_key_raw, payload)
    [trace] = _graph_traces(client, user_headers)

    markdown = _graph_markdown(client, user_headers, trace["id"])

    header, _gateway, inferer = _split_nodes(markdown)
    assert "**Status:** failed\n\n" in header
    assert f"**Failure Reason:** {trace['failure_reason']}\n\n" in header
    assert "**Answer:** none\n" in header
    assert inferer == (
        "## Node 1: inferer\n\n"
        "**Agent:** unknown\n\n"
        f"**Entered By:** edge {trace['node_traces'][1]['entry_edge_id']}\n\n"
        '**Condition:** `{"field": "ui_exception", "op": "eq", "value": true}`\n\n'
        "**Status:** cancelled\n\n"
        f"**Error:** {trace['node_traces'][1]['error']}\n\n"
        "_No agent trace recorded._\n"
    )
