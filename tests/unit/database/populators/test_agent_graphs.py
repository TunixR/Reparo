"""The agent graph populator: the UI Recovery graph, and the migration of old databases."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers
from sqlalchemy import Engine
from sqlmodel import Session, select

from database.agent_graphs.models import AgentGraph
from database.agents.models import Agent, SubAgent
from database.agents.specs import OutputModelSpec
from database.populators.agent_graphs import populate_agent_graphs
from tests.unit.fixtures.agent_graph_fixtures import MakeAgentGraph, Node

UI_EXCEPTION = {"field": "ui_exception", "op": "eq", "value": True}
RESPONSE_TO_RPA = {
    "name": "ResponseToRPA",
    "description": "The recovery outcome sent to the robot.",
    "fields": [
        {
            "name": "success",
            "type": "bool",
            "description": "Whether the recovery succeeded and the RPA can continue.",
        }
    ],
}
"""The gateway's output model before the UI Recovery graph."""


def _graphs(session: Session) -> dict[str, AgentGraph]:
    session.expire_all()
    return {graph.name: graph for graph in session.exec(select(AgentGraph))}


def _shape(graph: AgentGraph) -> dict[str, object]:
    """The graph's nodes and edges by node key and agent name."""
    keys = {node.id: node.key for node in graph.nodes}
    return {
        "nodes": sorted(
            (node.key, node.agent.name, node.is_entry, node.input_map)
            for node in graph.nodes
        ),
        "edges": sorted(
            (
                keys[edge.source_node_id],
                keys[edge.target_node_id],
                edge.condition.model_dump(mode="json", by_alias=True)
                if edge.condition is not None
                else None,
            )
            for edge in graph.edges
        ),
    }


def test_a_fresh_database_recovers_through_the_enabled_ui_recovery_graph(
    engine: Engine, session: Session, seeded_agents: dict[str, Agent]
):
    populate_agent_graphs(engine)

    [graph] = _graphs(session).values()
    assert graph.name == "UI Recovery"
    assert graph.enabled and graph.is_default_recovery
    assert graph.max_node_executions == 4
    assert graph.execution_timeout_s is None and graph.node_timeout_s is None
    assert graph.flags == []
    assert _shape(graph) == {
        "nodes": [
            (
                "continuation_inferer",
                "Continuation Inferer",
                False,
                {
                    "gui_summary": "tooled_guiagent.summary",
                    "goal_reached": "tooled_guiagent.goal_reached",
                },
            ),
            ("gateway", "Gateway Orchestrator", True, {}),
            ("recovery_inferer", "Recovery Inferer", False, {}),
            (
                "tooled_guiagent",
                "Tooled GuiAgent",
                False,
                {"task": "recovery_inferer.recovery_task"},
            ),
        ],
        "edges": [
            ("gateway", "recovery_inferer", UI_EXCEPTION),
            ("recovery_inferer", "tooled_guiagent", None),
            ("tooled_guiagent", "continuation_inferer", None),
        ],
    }


def test_an_existing_default_recovery_graph_keeps_recoveries(
    engine: Engine,
    session: Session,
    seeded_agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
):
    custom = make_agent_graph(
        Node("entry", seeded_agents["Recovery Inferer"]), name="Custom Recovery"
    )

    assert populate_agent_graphs(engine) == []

    graphs = _graphs(session)
    assert AgentGraph.default_recovery_id(session) == custom.id
    assert not graphs["UI Recovery"].enabled


@pytest.fixture
def old_database(session: Session, seeded_agents: dict[str, Agent]) -> dict[str, Agent]:
    """Agents as a database seeded before the UI Recovery graph has them.

    The agent populator adds the new agents but never overwrites one, so the gateway
    keeps its old prompt and output model and the UI Exception Handler stays enabled.
    """
    gateway = seeded_agents["Gateway Orchestrator"]
    gateway.prompt = "Route the failure to the UI Exception Handler."
    gateway.output_model = OutputModelSpec.model_validate(RESPONSE_TO_RPA)
    handler = seeded_agents["UI Exception Handler"]
    handler.enabled = True
    session.add_all([gateway, handler])
    session.add(
        SubAgent(parent_agent_id=gateway.id, child_agent_id=handler.id, limit=1)
    )
    session.commit()
    return seeded_agents


def test_an_old_database_gets_a_disabled_graph_and_the_steps_to_migrate(
    engine: Engine,
    session: Session,
    old_database: dict[str, Agent],
    capsys: pytest.CaptureFixture[str],
):
    steps = populate_agent_graphs(engine)

    graphs = _graphs(session)
    assert sorted(graphs) == ["Default Recovery", "UI Recovery"]
    assert not graphs["UI Recovery"].enabled
    assert not graphs["UI Recovery"].is_default_recovery
    legacy = graphs["Default Recovery"]
    assert legacy.enabled and legacy.is_default_recovery
    assert [(n.key, n.agent.name) for n in legacy.nodes] == [
        ("gateway", "Gateway Orchestrator")
    ]
    gateway = old_database["Gateway Orchestrator"]
    handler = old_database["UI Exception Handler"]
    assert [(s.method, s.path) for s in steps] == [
        ("PATCH", f"/agent_graphs/{legacy.id}"),
        ("PATCH", f"/agents/{gateway.id}"),
        ("PATCH", f"/agents/{handler.id}"),
        ("PATCH", f"/agent_graphs/{graphs['UI Recovery'].id}"),
    ]
    assert steps[1].body["output_model"]["name"] == "GatewayDecision"
    assert steps[1].body["prompt"].startswith("You are the Gateway Orchestrator")
    printed = capsys.readouterr().out
    assert "recoveries run that agent alone" in printed
    for number, step in enumerate(steps, start=1):
        assert f"  {number}. {step.action}:\n     PATCH {step.path} " in printed
    assert "<the text of database/populators/default_prompts/gateway.md>" in printed


def _dump(output_model: OutputModelSpec | None) -> str | None:
    return None if output_model is None else output_model.model_dump_json()


def _snapshot(session: Session) -> dict[str, object]:
    session.expire_all()
    agents = session.exec(select(Agent)).unique()
    return {
        "graphs": {
            name: (graph.id, graph.enabled, graph.is_default_recovery, _shape(graph))
            for name, graph in _graphs(session).items()
        },
        "agents": sorted(
            (a.name, a.enabled, _dump(a.output_model), a.prompt) for a in agents
        ),
    }


@pytest.mark.parametrize("database", ["seeded_agents", "old_database"])
def test_a_second_run_changes_nothing(
    database: str,
    engine: Engine,
    session: Session,
    request: pytest.FixtureRequest,
):
    _ = request.getfixturevalue(database)
    first_steps = populate_agent_graphs(engine)
    before = _snapshot(session)

    second_steps = populate_agent_graphs(engine)

    assert _snapshot(session) == before
    assert second_steps == first_steps


def test_following_the_steps_through_the_api_moves_recoveries_to_the_graph(
    engine: Engine,
    session: Session,
    old_database: dict[str, Agent],
    client: TestClient,
    admin_headers: Headers,
):
    steps = populate_agent_graphs(engine)

    for step in steps:
        response = client.request(
            step.method, step.path, json=step.body, headers=admin_headers
        )
        assert response.status_code == status.HTTP_200_OK, (step.action, response.text)

    graphs = _graphs(session)
    assert AgentGraph.default_recovery_id(session) == graphs["UI Recovery"].id
    assert graphs["UI Recovery"].flags == []
    assert not graphs["Default Recovery"].enabled
    assert not old_database["UI Exception Handler"].enabled
    assert populate_agent_graphs(engine) == []


def test_a_rerun_halfway_through_the_migration_prints_the_steps_left(
    engine: Engine,
    session: Session,
    old_database: dict[str, Agent],
    client: TestClient,
    admin_headers: Headers,
    capsys: pytest.CaptureFixture[str],
):
    *first_steps, enable = populate_agent_graphs(engine)
    for step in first_steps:
        response = client.request(
            step.method, step.path, json=step.body, headers=admin_headers
        )
        assert response.status_code == status.HTTP_200_OK, (step.action, response.text)
    _ = capsys.readouterr()

    assert populate_agent_graphs(engine) == [enable]

    assert "  1. " + str(enable) in capsys.readouterr().out
    assert AgentGraph.default_recovery_id(session) is None
