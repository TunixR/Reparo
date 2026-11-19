"""Administrators manage agent graphs through the API; enabled graphs stay valid."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers
from httpx2 import Response
from sqlmodel import Session, select

from database.agent_graphs import service
from database.agent_graphs.models import AgentGraphEdge, AgentGraphNode
from database.agent_graphs.runner import NodeRun
from database.agent_graphs.service import MAX_NODE_EXECUTIONS
from database.agents.models import Agent
from database.provider.models import Router
from tests.unit.fixtures.agent_fixtures import ScriptedAgent, recovery_arguments
from tests.unit.fixtures.agent_graph_fixtures import (
    GRAPH_REPORT,
    BuildAgentGraph,
    MakeAgentGraph,
    Node,
    add_edge_via_api,
    argument,
    create_graph_via_api,
    enable_via_api,
)
from tests.unit.shared.scripted_model import ToolCall


def test_an_administrator_creates_reads_updates_and_deletes_a_graph(
    client: TestClient, admin_headers: Headers, user_headers: Headers
):
    created = create_graph_via_api(
        client, admin_headers, name="Pipeline", max_node_executions=3
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    graph = created.json()
    assert graph["name"] == "Pipeline"
    assert graph["enabled"] is False
    assert graph["max_node_executions"] == 3
    assert graph["nodes"] == []
    assert graph["edges"] == []

    listed = client.get("/agent_graphs/", headers=user_headers)
    assert listed.status_code == status.HTTP_200_OK
    assert [g["id"] for g in listed.json()] == [graph["id"]]
    got = client.get(f"/agent_graphs/{graph['id']}", headers=user_headers)
    assert got.status_code == status.HTTP_200_OK
    assert got.json()["name"] == "Pipeline"

    updated = client.patch(
        f"/agent_graphs/{graph['id']}",
        json={"description": "Changed.", "max_node_executions": None},
        headers=admin_headers,
    )
    assert updated.status_code == status.HTTP_200_OK, updated.text
    assert updated.json()["description"] == "Changed."
    assert updated.json()["max_node_executions"] is None

    deleted = client.delete(f"/agent_graphs/{graph['id']}", headers=admin_headers)
    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    missing = client.get(f"/agent_graphs/{graph['id']}", headers=user_headers)
    assert missing.status_code == status.HTTP_404_NOT_FOUND


def test_reads_require_a_signed_in_user(client: TestClient):
    assert client.get("/agent_graphs/").status_code == status.HTTP_401_UNAUTHORIZED


def test_writes_are_rejected_for_non_administrators(
    client: TestClient,
    admin_headers: Headers,
    user_headers: Headers,
    graph_agents: dict[str, Agent],
):
    agent = graph_agents["Alpha"]
    graph_id = create_graph_via_api(client, admin_headers).json()["id"]
    node = client.post(
        f"/agent_graphs/{graph_id}/nodes",
        json={"key": "a", "agent_id": str(agent.id), "is_entry": True},
        headers=admin_headers,
    ).json()
    edge_body = {"source_node_id": node["id"], "target_node_id": node["id"]}
    attempts = [
        create_graph_via_api(client, user_headers),
        client.patch(
            f"/agent_graphs/{graph_id}", json={"enabled": True}, headers=user_headers
        ),
        client.delete(f"/agent_graphs/{graph_id}", headers=user_headers),
        client.post(
            f"/agent_graphs/{graph_id}/nodes",
            json={"key": "b", "agent_id": str(agent.id)},
            headers=user_headers,
        ),
        client.patch(
            f"/agent_graphs/{graph_id}/nodes/{node['id']}",
            json={"key": "c"},
            headers=user_headers,
        ),
        client.delete(
            f"/agent_graphs/{graph_id}/nodes/{node['id']}", headers=user_headers
        ),
        client.post(
            f"/agent_graphs/{graph_id}/edges", json=edge_body, headers=user_headers
        ),
    ]
    assert [r.status_code for r in attempts] == [status.HTTP_403_FORBIDDEN] * len(
        attempts
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"max_node_executions": 0},
        {"execution_timeout_s": -1},
        {"node_timeout_s": 0},
    ],
)
def test_limits_must_be_positive(
    client: TestClient, admin_headers: Headers, fields: dict[str, Any]
):
    response = create_graph_via_api(client, admin_headers, **fields)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_limits_accept_null(client: TestClient, admin_headers: Headers):
    response = create_graph_via_api(
        client,
        admin_headers,
        max_node_executions=None,
        execution_timeout_s=None,
        node_timeout_s=None,
    )
    assert response.status_code == status.HTTP_201_CREATED, response.text


def _rejected(response: Response, *fragments: str) -> None:
    """The write was rejected with 400, and one violation says all ``fragments``."""
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
    violations: list[str] = response.json()["detail"]["violations"]
    assert any(all(f in v for f in fragments) for v in violations), violations


CONDITION = {"field": "done", "op": "eq", "value": True}


def test_a_draft_accepts_an_invalid_structure_and_enabling_lists_the_violations(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b, True), ("c", c)], [("a", "c")])

    response = enable_via_api(client, admin_headers, graph.id)

    _rejected(response, "2 entry nodes", "'a'", "'b'")
    assert (
        client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()["enabled"]
        is False
    )


def test_an_enabled_graph_can_always_be_disabled(
    client: TestClient,
    admin_headers: Headers,
    session: Session,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    alpha = graph_agents["Alpha"]
    graph = build_agent_graph([("a", alpha)], enabled=True)
    alpha.enabled = False
    session.add(alpha)
    session.commit()

    response = client.patch(
        f"/agent_graphs/{graph.id}", json={"enabled": False}, headers=admin_headers
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["enabled"] is False


def test_enabling_requires_every_node_agent_to_be_enabled(
    client: TestClient,
    admin_headers: Headers,
    disabled_agent: Agent,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph(
        [("a", graph_agents["Alpha"]), ("b", disabled_agent)], [("a", "b")]
    )

    _rejected(enable_via_api(client, admin_headers, graph.id), "'Dormant'", "Node 'b'")


def test_a_new_node_must_keep_an_enabled_graph_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph([("a", graph_agents["Alpha"])], enabled=True)

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "b", "agent_id": str(graph_agents["Beta"].id)},
        headers=admin_headers,
    )

    _rejected(response, "'b'", "unreachable")
    nodes = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()[
        "nodes"
    ]
    assert [n["key"] for n in nodes] == ["a"]


def test_a_node_of_an_enabled_graph_needs_an_enabled_agent(
    client: TestClient,
    admin_headers: Headers,
    disabled_agent: Agent,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph([("a", graph_agents["Alpha"])], enabled=True)

    response = client.patch(
        f"/agent_graphs/{graph.id}/nodes/{graph.nodes['a']}",
        json={"agent_id": str(disabled_agent.id)},
        headers=admin_headers,
    )

    _rejected(response, "'Dormant'", "Node 'a'")


def test_removing_an_edge_must_keep_an_enabled_graph_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, _ = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b")], enabled=True)
    edge_id = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()[
        "edges"
    ][0]["id"]

    response = client.delete(
        f"/agent_graphs/{graph.id}/edges/{edge_id}", headers=admin_headers
    )

    _rejected(response, "'b'", "unreachable")


def test_a_draft_accepts_any_removal(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, _ = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b")])

    response = client.delete(
        f"/agent_graphs/{graph.id}/nodes/{graph.nodes['a']}", headers=admin_headers
    )

    assert response.status_code == status.HTTP_204_NO_CONTENT
    body = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()
    assert [n["key"] for n in body["nodes"]] == ["b"]
    assert body["edges"] == []


def test_deleting_a_graph_deletes_its_nodes_and_edges(
    client: TestClient,
    admin_headers: Headers,
    session: Session,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, _ = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b")], enabled=True)

    response = client.delete(f"/agent_graphs/{graph.id}", headers=admin_headers)

    assert response.status_code == status.HTTP_204_NO_CONTENT
    session.expire_all()
    assert session.exec(select(AgentGraphNode)).all() == []
    assert session.exec(select(AgentGraphEdge)).all() == []
    assert session.get(Agent, a.id) is not None


def test_the_entry_node_has_no_incoming_edges(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, _ = graph_agents.values()
    graph = build_agent_graph(
        [("a", a), ("b", b)], [("a", "b"), ("b", "a")], max_node_executions=4
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id), "entry node 'a'", "incoming"
    )


def test_a_graph_needs_an_entry_node(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph([("a", graph_agents["Alpha"], False)])

    _rejected(enable_via_api(client, admin_headers, graph.id), "no entry node")


def test_every_node_is_reachable_from_the_entry(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b), ("c", c)], [("a", "b")])

    _rejected(
        enable_via_api(client, admin_headers, graph.id), "'c'", "unreachable", "'a'"
    )


def test_edges_are_unique_and_stay_inside_their_graph(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b), ("c", c)], [("a", "b", 0)])
    other = build_agent_graph([("x", a)])

    duplicate = add_edge_via_api(
        client, admin_headers, graph.id, graph.nodes, "a", "b", 1
    )
    same_priority = add_edge_via_api(
        client, admin_headers, graph.id, graph.nodes, "a", "c", 0
    )
    foreign = add_edge_via_api(
        client, admin_headers, graph.id, {**graph.nodes, **other.nodes}, "a", "x", 2
    )

    _rejected(duplicate, "already exists")
    _rejected(same_priority, "priority 0")
    _rejected(foreign, "inside one graph")


def test_a_node_has_at_most_one_else_edge(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph(
        [("a", a), ("b", b), ("c", c)], [("a", "b", 0), ("a", "c", 1)]
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id), "Node 'a'", "2 unconditional"
    )


def test_the_else_edge_is_evaluated_last(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph(
        [("a", a), ("b", b), ("c", c)], [("a", "b", 0), ("a", "c", 1, CONDITION)]
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id), "node 'a'", "lowest priority"
    )


def test_conditional_edges_before_the_else_edge_are_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph(
        [("a", a), ("b", b), ("c", c)], [("a", "b", 0, CONDITION), ("a", "c", 1)]
    )

    assert (
        enable_via_api(client, admin_headers, graph.id).status_code
        == status.HTTP_200_OK
    )


def test_a_cycle_needs_max_node_executions(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph(
        [("a", a), ("b", b), ("c", c)], [("a", "b"), ("b", "c"), ("c", "b")]
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id), "cycle", "max_node_executions"
    )
    bounded = client.patch(
        f"/agent_graphs/{graph.id}",
        json={"enabled": True, "max_node_executions": 6},
        headers=admin_headers,
    )
    assert bounded.status_code == status.HTTP_200_OK, bounded.text


def test_max_node_executions_has_a_configured_maximum(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph(
        [("a", graph_agents["Alpha"])], max_node_executions=MAX_NODE_EXECUTIONS + 1
    )

    _rejected(enable_via_api(client, admin_headers, graph.id), str(MAX_NODE_EXECUTIONS))


def test_a_gateway_is_only_an_entry_node(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    graphless_gateway: Agent,
):
    graph = build_agent_graph(
        [("a", graph_agents["Alpha"]), ("gateway", graphless_gateway)],
        [("a", "gateway")],
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id), "'Gateway'", "entry node"
    )


def test_a_gateway_enters_at_most_one_enabled_graph(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graphless_gateway: Agent,
):
    first = build_agent_graph(
        [("gateway", graphless_gateway)], name="First", enabled=True
    )
    second = build_agent_graph([("gateway", graphless_gateway)], name="Second")

    _rejected(enable_via_api(client, admin_headers, second.id), "'Gateway'", "'First'")
    assert first.id != second.id


def test_one_agent_may_back_several_nodes_with_different_keys(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    alpha = graph_agents["Alpha"]
    graph = build_agent_graph(
        [("first", alpha), ("again", alpha)], [("first", "again")]
    )

    assert (
        enable_via_api(client, admin_headers, graph.id).status_code
        == status.HTTP_200_OK
    )


def test_a_node_key_is_unique_in_its_graph(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph([("a", graph_agents["Alpha"])])

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "a", "agent_id": str(graph_agents["Beta"].id)},
        headers=admin_headers,
    )

    _rejected(response, "already has a node 'a'")


def test_a_node_needs_an_existing_agent(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    graph = build_agent_graph([("a", graph_agents["Alpha"])])

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "b", "agent_id": str(uuid.uuid4())},
        headers=admin_headers,
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_a_graph_created_and_enabled_through_the_api_is_the_one_recoveries_run(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    scripted_agent: ScriptedAgent,
    make_agent_graph: MakeAgentGraph,
    make_robot_key: Callable[..., Any],
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    _ = make_agent_graph(Node("old", graph_agents["Alpha"]), name="Old Default")
    worker, model = scripted_agent(
        "Working.", ToolCall("ScriptedReport", {"done": True})
    )
    ran: list[str] = []
    original = Agent.run_node

    async def _spy(
        self: Agent, invocation_state: dict[str, Any], input_map: Mapping[str, str]
    ) -> NodeRun:
        ran.append(self.name)
        return await original(self, invocation_state, input_map)

    monkeypatch.setattr(Agent, "run_node", _spy)
    graph_id = create_graph_via_api(
        client, admin_headers, name="New Default", is_default_recovery=True
    ).json()["id"]
    node = client.post(
        f"/agent_graphs/{graph_id}/nodes",
        json={
            "key": "worker",
            "agent_id": str(worker.id),
            "is_entry": True,
            "input_map": {"task": "recovery.task_name"},
        },
        headers=admin_headers,
    )
    assert node.status_code == status.HTTP_201_CREATED, node.text
    assert (
        enable_via_api(client, admin_headers, graph_id).status_code
        == status.HTTP_200_OK
    )
    key_raw = "test-robot-key-graph-api-0001"
    _ = make_robot_key(name="Graph API", enabled=True, key_raw=key_raw)

    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": key_raw}
    ) as ws:
        ws.send_json(valid_recovery_payload)
        message = ws.receive_json()
        while message["type"] == "ping":
            message = ws.receive_json()

    assert message["type"] == "done", message
    assert ran == [worker.name]
    assert len(model.requests) == 2
    graphs = {
        g["name"]: g for g in client.get("/agent_graphs/", headers=admin_headers).json()
    }
    assert graphs["New Default"]["is_default_recovery"] is True
    assert graphs["Old Default"]["is_default_recovery"] is False


def test_an_edge_condition_round_trips_and_a_malformed_one_is_rejected(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b), ("c", c)], [("a", "b", 0)])
    negated = {"not": {"field": "done", "op": "eq", "value": True}}

    added = add_edge_via_api(
        client, admin_headers, graph.id, graph.nodes, "a", "c", 1, negated
    )
    malformed = add_edge_via_api(
        client,
        admin_headers,
        graph.id,
        graph.nodes,
        "a",
        "c",
        1,
        {"field": "done", "op": "is"},
    )
    changed = client.patch(
        f"/agent_graphs/{graph.id}/edges/{added.json()['id']}",
        json={"condition": None, "priority": 2},
        headers=admin_headers,
    )

    assert added.status_code == status.HTTP_201_CREATED, added.text
    assert added.json()["condition"] == negated
    assert malformed.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert changed.status_code == status.HTTP_200_OK, changed.text
    assert changed.json()["condition"] is None
    assert changed.json()["priority"] == 2


def test_changing_an_edge_must_keep_an_enabled_graph_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, _ = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b")], enabled=True)
    edge_id = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()[
        "edges"
    ][0]["id"]

    response = client.patch(
        f"/agent_graphs/{graph.id}/edges/{edge_id}",
        json={"target_node_id": graph.nodes["a"]},
        headers=admin_headers,
    )

    _rejected(response, "entry node 'a'", "incoming")


def test_a_write_only_checks_the_agent_of_the_node_it_writes(
    client: TestClient,
    admin_headers: Headers,
    session: Session,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b")], enabled=True)
    b.enabled = False
    session.add(b)
    session.commit()

    renamed = client.patch(
        f"/agent_graphs/{graph.id}",
        json={"description": "Renamed."},
        headers=admin_headers,
    )
    added = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "c", "agent_id": str(c.id)},
        headers=admin_headers,
    )

    assert renamed.status_code == status.HTTP_200_OK, renamed.text
    _rejected(added, "'c'")
    assert "Beta" not in added.text


def test_a_write_that_loses_a_race_on_a_unique_constraint_is_rejected(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    monkeypatch: pytest.MonkeyPatch,
):
    graph = build_agent_graph([("a", graph_agents["Alpha"])])
    # A concurrent request adding the same key passes the check before this one
    # commits, so only the database's unique constraint sees the duplicate.
    monkeypatch.setattr(service, "_ensure_unique_key", lambda *_args: None)

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "a", "agent_id": str(graph_agents["Beta"].id)},
        headers=admin_headers,
    )

    _rejected(response, "concurrent change", "reload the graph and retry")
    nodes = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()
    assert [n["key"] for n in nodes["nodes"]] == ["a"]


def test_a_condition_compares_a_field_of_the_source_output(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, c = graph_agents.values()
    typo = {"field": "finding.raeson", "op": "eq", "value": "timeout"}
    graph = build_agent_graph(
        [("a", a), ("b", b), ("c", c)], [("a", "b", 0, typo), ("a", "c", 1)]
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id),
        "'a' → 'b'",
        "'raeson'",
        "Finding",
        "reason",
    )


@pytest.mark.parametrize(
    ("condition", "fragments"),
    [
        ({"field": "attempts", "op": "eq", "value": "three"}, ("'attempts'", "three")),
        ({"field": "attempts", "op": "in", "value": [1, "x"]}, ("'attempts'", '"x"')),
        ({"field": "done", "op": "lt", "value": True}, ("'done'", "'lt'", "order")),
        ({"field": "done", "op": "is_null"}, ("'done'", "'is_null'", "never null")),
        (
            {"field": "finding.tags.x", "op": "eq", "value": 1},
            ("finding.tags", "no fields"),
        ),
        (
            {"not": {"any": [{"field": "extra.k", "op": "ge", "value": "high"}]}},
            ("'extra.k'", "high"),
        ),
    ],
)
def test_a_condition_operator_and_value_fit_the_field_type(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    condition: dict[str, Any],
    fragments: tuple[str, ...],
):
    a, b, _ = graph_agents.values()
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b", 0, condition)])

    _rejected(enable_via_api(client, admin_headers, graph.id), "'a' → 'b'", *fragments)


def test_conditions_that_fit_the_output_model_are_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
):
    a, b, _ = graph_agents.values()
    condition = {
        "all": [
            {"field": "done", "op": "eq", "value": True},
            {"field": "attempts", "op": "le", "value": 3},
            {"field": "note", "op": "not_null"},
            {"field": "finding.reason", "op": "in", "value": ["timeout", "popup"]},
            {"field": "extra.retries", "op": "gt", "value": 0},
        ]
    }
    graph = build_agent_graph([("a", a), ("b", b)], [("a", "b", 0, condition)])

    response = enable_via_api(client, admin_headers, graph.id)

    assert response.status_code == status.HTTP_200_OK, response.text


def test_a_node_with_conditional_edges_needs_an_output_model(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    mock_agent: Agent,
):
    graph = build_agent_graph(
        [("free", mock_agent), ("b", graph_agents["Beta"])],
        [("free", "b", 0, CONDITION)],
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id),
        "Node 'free'",
        "conditional",
        "no output model",
    )


def test_a_gui_node_is_routed_on_its_run_report(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    gui_agent: Agent,
):
    nodes = [
        ("start", graph_agents["Alpha"]),
        ("gui", gui_agent, False, {"task": "start.finding.reason"}),
        ("b", graph_agents["Beta"]),
    ]
    on_success = {"field": "success", "op": "eq", "value": True}
    on_goal = {"field": "goal_reached", "op": "eq", "value": True}
    wrong = build_agent_graph(nodes, [("start", "gui"), ("gui", "b", 0, on_success)])
    right = build_agent_graph(nodes, [("start", "gui"), ("gui", "b", 0, on_goal)])

    _rejected(
        enable_via_api(client, admin_headers, wrong.id),
        "'gui' → 'b'",
        "'success'",
        "GuiRunReport",
    )
    assert (
        enable_via_api(client, admin_headers, right.id).status_code
        == status.HTTP_200_OK
    )


@pytest.mark.parametrize(
    ("input_map", "fragments"),
    [
        ({"tsk": "recovery.task_name"}, ("'tsk'", "isn't an argument", "task")),
        ({"task": "recovery.task"}, ("'recovery.task'", "'task'", "task_name")),
        (
            {"task": "recovery.ui_log.case_id"},
            ("'recovery.ui_log.case_id'", "no fields"),
        ),
        ({"task": "ghost.reason"}, ("'ghost.reason'", "no node 'ghost'")),
        ({"task": "a.finding.raeson"}, ("'a.finding.raeson'", "'raeson'", "Finding")),
        (
            {"task": "task_name"},
            ("'task_name'", "recovery.<field>", "<node_key>.<field>"),
        ),
    ],
)
def test_an_input_map_maps_arguments_to_recovery_inputs_or_node_outputs(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    tasked_agent: Agent,
    input_map: dict[str, str],
    fragments: tuple[str, ...],
):
    graph = build_agent_graph(
        [("a", graph_agents["Alpha"]), ("b", tasked_agent, False, input_map)],
        [("a", "b")],
    )

    _rejected(enable_via_api(client, admin_headers, graph.id), "Node 'b'", *fragments)


@pytest.mark.parametrize(
    "path",
    ["recovery.task_name", "recovery.errored_act.error_code", "a.finding.reason"],
)
def test_an_input_map_that_resolves_is_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    tasked_agent: Agent,
    path: str,
):
    graph = build_agent_graph(
        [("a", graph_agents["Alpha"]), ("b", tasked_agent, False, {"task": path})],
        [("a", "b")],
    )

    response = enable_via_api(client, admin_headers, graph.id)

    assert response.status_code == status.HTTP_200_OK, response.text


GOAL_REACHED = {"field": "goal_reached", "op": "eq", "value": True}


@pytest.fixture
def noted_agent(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    """An agent taking a ``note``, a field of the `GraphReport` graph agents output."""
    return make_agent(
        router=mock_router,
        name="Noted",
        arguments=[argument("note")],
        response_model=GRAPH_REPORT,
    )


def test_an_argument_bound_only_on_one_branch_is_rejected(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    gui_agent: Agent,
    mock_agent: Agent,
    noted_agent: Agent,
    tasked_agent: Agent,
):
    gui = ("gui", gui_agent, True, {"task": "recovery.task_name"})
    branches = [("gui", "b", 0, GOAL_REACHED), ("gui", "c", 1), ("b", "d"), ("c", "d")]
    by_name = build_agent_graph(
        [gui, ("b", graph_agents["Alpha"]), ("c", mock_agent), ("d", noted_agent)],
        branches,
    )
    mapped = build_agent_graph(
        [
            gui,
            ("b", graph_agents["Alpha"]),
            ("c", mock_agent),
            ("d", tasked_agent, False, {"task": "b.finding.reason"}),
        ],
        branches,
    )

    _rejected(
        enable_via_api(client, admin_headers, by_name.id),
        "Node 'd'",
        "'note'",
        "'b'",
        "every path",
    )
    _rejected(
        enable_via_api(client, admin_headers, mapped.id),
        "Node 'd'",
        "'b.finding.reason'",
        "every path",
    )


def test_an_argument_must_bind(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    tasked_agent: Agent,
):
    graph = build_agent_graph(
        [("a", graph_agents["Alpha"]), ("b", tasked_agent)], [("a", "b")]
    )

    _rejected(
        enable_via_api(client, admin_headers, graph.id),
        "Node 'b'",
        "'task'",
        "input_map",
    )


def test_arguments_bind_from_recovery_inputs_and_from_nodes_on_every_path(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    gui_agent: Agent,
    mock_agent: Agent,
    noted_agent: Agent,
    gateway_like: Agent,
):
    graph = build_agent_graph(
        [
            ("start", gateway_like),
            ("a", graph_agents["Alpha"]),
            ("gui", gui_agent, False, {"task": "a.finding.reason"}),
            ("c", mock_agent),
            ("d", noted_agent),
        ],
        [
            ("start", "a"),
            ("a", "gui"),
            ("gui", "c", 0, GOAL_REACHED),
            ("gui", "d", 1),
            ("c", "d"),
        ],
    )

    response = enable_via_api(client, admin_headers, graph.id)

    assert response.status_code == status.HTTP_200_OK, response.text


@pytest.fixture
def gateway_like(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    """An agent taking the seven recovery inputs by name."""
    return make_agent(router=mock_router, name="Reader", arguments=recovery_arguments())


def test_a_node_that_can_end_the_graph_without_success_is_a_warning(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    gui_agent: Agent,
    graphless_gateway: Agent,
):
    on_done = {"field": "done", "op": "eq", "value": True}
    graph = build_agent_graph(
        [
            ("gateway", graphless_gateway),
            ("a", graph_agents["Alpha"]),
            ("gui", gui_agent, False, {"task": "a.finding.reason"}),
        ],
        [("gateway", "a"), ("a", "gui", 0, on_done)],
    )

    response = enable_via_api(client, admin_headers, graph.id)

    assert response.status_code == status.HTTP_200_OK, response.text
    warnings: list[str] = response.json()["warnings"]
    assert len(warnings) == 1, warnings
    assert all(f in warnings[0] for f in ("Node 'gui'", "'success'", "GuiRunReport"))
    got = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers)
    assert got.json()["warnings"] == warnings


def test_a_gateway_may_end_the_graph_without_success(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graphless_gateway: Agent,
):
    graph = build_agent_graph([("gateway", graphless_gateway)])

    response = enable_via_api(client, admin_headers, graph.id)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["warnings"] == []


def test_a_write_to_an_enabled_graph_must_keep_its_bindings_valid(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    tasked_agent: Agent,
):
    graph = build_agent_graph(
        [
            ("a", graph_agents["Alpha"]),
            ("b", tasked_agent, False, {"task": "recovery.task_name"}),
        ],
        [("a", "b")],
        enabled=True,
    )

    response = client.patch(
        f"/agent_graphs/{graph.id}/nodes/{graph.nodes['b']}",
        json={"input_map": {"task": "a.finding.raeson"}},
        headers=admin_headers,
    )

    _rejected(response, "Node 'b'", "'raeson'")
    body = client.get(f"/agent_graphs/{graph.id}", headers=admin_headers).json()
    assert body["nodes"][1]["input_map"] == {"task": "recovery.task_name"}


def test_a_recovery_path_reads_the_recovery_inputs_even_beside_a_node_keyed_recovery(
    client: TestClient,
    admin_headers: Headers,
    build_agent_graph: BuildAgentGraph,
    graph_agents: dict[str, Agent],
    tasked_agent: Agent,
):
    graph = build_agent_graph(
        [
            ("a", graph_agents["Alpha"]),
            ("recovery", graph_agents["Beta"]),
            ("b", tasked_agent, False, {"task": "recovery.task_name"}),
        ],
        [("a", "recovery", 0, CONDITION), ("a", "b", 1), ("recovery", "b")],
    )

    response = enable_via_api(client, admin_headers, graph.id)

    assert response.status_code == status.HTTP_200_OK, response.text
