"""Administrators manage agent graphs through the API; enabled graphs stay valid."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers
from httpx2 import Response
from sqlmodel import Session, select

from database.agent_graphs.models import AgentGraphEdge, AgentGraphNode
from database.agent_graphs.runner import NodeRun
from database.agent_graphs.service import MAX_NODE_EXECUTIONS
from database.agents.models import Agent, AgentType
from database.auth.models import User
from database.provider.models import Router
from tests.unit.fixtures.agent_fixtures import ScriptedAgent
from tests.unit.fixtures.agent_graph_fixtures import MakeAgentGraph, Node
from tests.unit.shared.auth_helpers import make_auth_headers
from tests.unit.shared.scripted_model import ToolCall


@pytest.fixture
def admin(session: Session, mock_admin: User) -> Headers:
    return make_auth_headers(mock_admin, session)


@pytest.fixture
def user(session: Session, mock_user: User) -> Headers:
    return make_auth_headers(mock_user, session)


@pytest.fixture
def agent(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    return make_agent(router=mock_router, name="Worker")


def _create_graph(client: TestClient, headers: Headers, **fields: Any) -> Response:
    body = {"name": f"Graph {uuid.uuid4()}", "description": "A test graph."} | fields
    return client.post("/agent_graphs/", json=body, headers=headers)


def test_an_administrator_creates_reads_updates_and_deletes_a_graph(
    client: TestClient, admin: Headers, user: Headers
):
    created = _create_graph(client, admin, name="Pipeline", max_node_executions=3)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    graph = created.json()
    assert graph["name"] == "Pipeline"
    assert graph["enabled"] is False
    assert graph["max_node_executions"] == 3
    assert graph["nodes"] == []
    assert graph["edges"] == []

    listed = client.get("/agent_graphs/", headers=user)
    assert listed.status_code == status.HTTP_200_OK
    assert [g["id"] for g in listed.json()] == [graph["id"]]
    got = client.get(f"/agent_graphs/{graph['id']}", headers=user)
    assert got.status_code == status.HTTP_200_OK
    assert got.json()["name"] == "Pipeline"

    updated = client.patch(
        f"/agent_graphs/{graph['id']}",
        json={"description": "Changed.", "max_node_executions": None},
        headers=admin,
    )
    assert updated.status_code == status.HTTP_200_OK, updated.text
    assert updated.json()["description"] == "Changed."
    assert updated.json()["max_node_executions"] is None

    deleted = client.delete(f"/agent_graphs/{graph['id']}", headers=admin)
    assert deleted.status_code == status.HTTP_204_NO_CONTENT
    missing = client.get(f"/agent_graphs/{graph['id']}", headers=user)
    assert missing.status_code == status.HTTP_404_NOT_FOUND


def test_reads_require_a_signed_in_user(client: TestClient):
    assert client.get("/agent_graphs/").status_code == status.HTTP_401_UNAUTHORIZED


def test_writes_are_rejected_for_non_administrators(
    client: TestClient, admin: Headers, user: Headers, agent: Agent
):
    graph_id = _create_graph(client, admin).json()["id"]
    node = client.post(
        f"/agent_graphs/{graph_id}/nodes",
        json={"key": "a", "agent_id": str(agent.id), "is_entry": True},
        headers=admin,
    ).json()
    edge_body = {"source_node_id": node["id"], "target_node_id": node["id"]}
    attempts = [
        _create_graph(client, user),
        client.patch(f"/agent_graphs/{graph_id}", json={"enabled": True}, headers=user),
        client.delete(f"/agent_graphs/{graph_id}", headers=user),
        client.post(
            f"/agent_graphs/{graph_id}/nodes",
            json={"key": "b", "agent_id": str(agent.id)},
            headers=user,
        ),
        client.patch(
            f"/agent_graphs/{graph_id}/nodes/{node['id']}",
            json={"key": "c"},
            headers=user,
        ),
        client.delete(f"/agent_graphs/{graph_id}/nodes/{node['id']}", headers=user),
        client.post(f"/agent_graphs/{graph_id}/edges", json=edge_body, headers=user),
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
    client: TestClient, admin: Headers, fields: dict[str, Any]
):
    response = _create_graph(client, admin, **fields)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_limits_accept_null(client: TestClient, admin: Headers):
    response = _create_graph(
        client,
        admin,
        max_node_executions=None,
        execution_timeout_s=None,
        node_timeout_s=None,
    )
    assert response.status_code == status.HTTP_201_CREATED, response.text


type NodeSpec = tuple[str, Agent] | tuple[str, Agent, bool]
type EdgeSpec = (
    tuple[str, str] | tuple[str, str, int] | tuple[str, str, int, dict[str, Any]]
)


class Built:
    """A graph made through the API: its id and its node ids by key."""

    def __init__(self, graph_id: str, nodes: dict[str, str]) -> None:
        self.id = graph_id
        self.nodes = nodes


@pytest.fixture
def build(client: TestClient, admin: Headers) -> Callable[..., Built]:
    """Build a graph through the API, enabled once complete if asked; the first node
    is the entry by default.

    Edges are ``(source, target[, priority[, condition]])``; without a priority,
    an edge's place in the list is its priority.
    """

    def _build(
        nodes: list[NodeSpec],
        edges: Sequence[EdgeSpec] = (),
        enabled: bool = False,
        **fields: Any,
    ) -> Built:
        created = _create_graph(client, admin, **fields)
        assert created.status_code == status.HTTP_201_CREATED, created.text
        graph_id = created.json()["id"]
        ids: dict[str, str] = {}
        for index, spec in enumerate(nodes):
            key, agent = spec[0], spec[1]
            is_entry = spec[2] if len(spec) == 3 else index == 0
            response = client.post(
                f"/agent_graphs/{graph_id}/nodes",
                json={"key": key, "agent_id": str(agent.id), "is_entry": is_entry},
                headers=admin,
            )
            assert response.status_code == status.HTTP_201_CREATED, response.text
            ids[key] = response.json()["id"]
        for index, spec in enumerate(edges):
            match spec:
                case (source, target):
                    priority, condition = index, None
                case (source, target, priority):
                    condition = None
                case (source, target, priority, condition):
                    pass
            response = _add_edge(
                client, admin, graph_id, ids, source, target, priority, condition
            )
            assert response.status_code == status.HTTP_201_CREATED, response.text
        if enabled:
            response = _enable(client, admin, graph_id)
            assert response.status_code == status.HTTP_200_OK, response.text
        return Built(graph_id, ids)

    return _build


def _add_edge(
    client: TestClient,
    headers: Headers,
    graph_id: str,
    ids: dict[str, str],
    source: str,
    target: str,
    priority: int = 0,
    condition: dict[str, Any] | None = None,
) -> Response:
    return client.post(
        f"/agent_graphs/{graph_id}/edges",
        json={
            "source_node_id": ids[source],
            "target_node_id": ids[target],
            "priority": priority,
            "condition": condition,
        },
        headers=headers,
    )


def _enable(client: TestClient, headers: Headers, graph_id: str) -> Response:
    return client.patch(
        f"/agent_graphs/{graph_id}", json={"enabled": True}, headers=headers
    )


def _rejected(response: Response, rule: str, *fragments: str) -> None:
    """The write was rejected with 409 for ``rule``, saying ``fragments``."""
    assert response.status_code == status.HTTP_409_CONFLICT, response.text
    messages = [v["message"] for v in response.json()["detail"]["violations"]]
    rules = [v["rule"] for v in response.json()["detail"]["violations"]]
    assert rule in rules, response.text
    message = messages[rules.index(rule)]
    for fragment in fragments:
        assert fragment in message, message


@pytest.fixture
def agents(make_agent: Callable[..., Agent], mock_router: Router) -> dict[str, Agent]:
    return {
        name: make_agent(router=mock_router, name=name)
        for name in ("Alpha", "Beta", "Gamma")
    }


@pytest.fixture
def dormant(
    session: Session, make_agent: Callable[..., Agent], mock_router: Router
) -> Agent:
    agent = make_agent(router=mock_router, name="Dormant")
    agent.enabled = False
    session.add(agent)
    session.commit()
    return agent


@pytest.fixture
def gateway(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    return make_agent(router=mock_router, name="Gateway", type=AgentType.GatewayAgent)


CONDITION = {"field": "done", "op": "eq", "value": True}


def test_a_draft_accepts_an_invalid_structure_and_enabling_lists_the_violations(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b, True), ("c", c)], [("a", "c")])

    response = _enable(client, admin, graph.id)

    _rejected(response, "S1", "2 entry nodes", "'a'", "'b'")
    assert (
        client.get(f"/agent_graphs/{graph.id}", headers=admin).json()["enabled"]
        is False
    )


def test_v3_an_enabled_graph_can_always_be_disabled(
    client: TestClient,
    admin: Headers,
    session: Session,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    alpha = agents["Alpha"]
    graph = build([("a", alpha)], enabled=True)
    alpha.enabled = False
    session.add(alpha)
    session.commit()

    response = client.patch(
        f"/agent_graphs/{graph.id}", json={"enabled": False}, headers=admin
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["enabled"] is False


def test_v4_enabling_requires_every_node_agent_to_be_enabled(
    client: TestClient,
    admin: Headers,
    dormant: Agent,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"]), ("b", dormant)], [("a", "b")])

    _rejected(_enable(client, admin, graph.id), "V4", "'Dormant'", "Node 'b'")


def test_v5_a_new_node_must_keep_an_enabled_graph_valid(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"])], enabled=True)

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "b", "agent_id": str(agents["Beta"].id)},
        headers=admin,
    )

    _rejected(response, "S2", "'b'", "unreachable")
    nodes = client.get(f"/agent_graphs/{graph.id}", headers=admin).json()["nodes"]
    assert [n["key"] for n in nodes] == ["a"]


def test_v5_a_node_of_an_enabled_graph_needs_an_enabled_agent(
    client: TestClient,
    admin: Headers,
    dormant: Agent,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"])], enabled=True)

    response = client.patch(
        f"/agent_graphs/{graph.id}/nodes/{graph.nodes['a']}",
        json={"agent_id": str(dormant.id)},
        headers=admin,
    )

    _rejected(response, "V5", "'Dormant'", "Node 'a'")


def test_v6_removing_an_edge_must_keep_an_enabled_graph_valid(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, _ = agents.values()
    graph = build([("a", a), ("b", b)], [("a", "b")], enabled=True)
    edge_id = client.get(f"/agent_graphs/{graph.id}", headers=admin).json()["edges"][0][
        "id"
    ]

    response = client.delete(f"/agent_graphs/{graph.id}/edges/{edge_id}", headers=admin)

    _rejected(response, "S2", "'b'", "unreachable")


def test_v6_a_draft_accepts_any_removal(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, _ = agents.values()
    graph = build([("a", a), ("b", b)], [("a", "b")])

    response = client.delete(
        f"/agent_graphs/{graph.id}/nodes/{graph.nodes['a']}", headers=admin
    )

    assert response.status_code == status.HTTP_204_NO_CONTENT
    body = client.get(f"/agent_graphs/{graph.id}", headers=admin).json()
    assert [n["key"] for n in body["nodes"]] == ["b"]
    assert body["edges"] == []


def test_v8_deleting_a_graph_deletes_its_nodes_and_edges(
    client: TestClient,
    admin: Headers,
    session: Session,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, _ = agents.values()
    graph = build([("a", a), ("b", b)], [("a", "b")], enabled=True)

    response = client.delete(f"/agent_graphs/{graph.id}", headers=admin)

    assert response.status_code == status.HTTP_204_NO_CONTENT
    session.expire_all()
    assert session.exec(select(AgentGraphNode)).all() == []
    assert session.exec(select(AgentGraphEdge)).all() == []
    assert session.get(Agent, a.id) is not None


def test_s1_the_entry_node_has_no_incoming_edges(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, _ = agents.values()
    graph = build([("a", a), ("b", b)], [("a", "b"), ("b", "a")], max_node_executions=4)

    _rejected(_enable(client, admin, graph.id), "S1", "entry node 'a'", "incoming")


def test_s1_a_graph_needs_an_entry_node(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"], False)])

    _rejected(_enable(client, admin, graph.id), "S1", "no entry node")


def test_s2_every_node_is_reachable_from_the_entry(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b), ("c", c)], [("a", "b")])

    _rejected(_enable(client, admin, graph.id), "S2", "'c'", "unreachable", "'a'")


def test_s3_edges_are_unique_and_stay_inside_their_graph(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b), ("c", c)], [("a", "b", 0)])
    other = build([("x", a)])

    duplicate = _add_edge(client, admin, graph.id, graph.nodes, "a", "b", 1)
    same_priority = _add_edge(client, admin, graph.id, graph.nodes, "a", "c", 0)
    foreign = _add_edge(client, admin, graph.id, graph.nodes | other.nodes, "a", "x", 2)

    _rejected(duplicate, "S3", "already exists")
    _rejected(same_priority, "S3", "priority 0")
    _rejected(foreign, "S3", "inside one graph")


def test_s4_a_node_has_at_most_one_else_edge(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b), ("c", c)], [("a", "b", 0), ("a", "c", 1)])

    _rejected(_enable(client, admin, graph.id), "S4", "Node 'a'", "2 unconditional")


def test_s4_the_else_edge_is_evaluated_last(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build(
        [("a", a), ("b", b), ("c", c)], [("a", "b", 0), ("a", "c", 1, CONDITION)]
    )

    _rejected(_enable(client, admin, graph.id), "S4", "node 'a'", "lowest priority")


def test_s4_conditional_edges_before_the_else_edge_are_valid(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build(
        [("a", a), ("b", b), ("c", c)], [("a", "b", 0, CONDITION), ("a", "c", 1)]
    )

    assert _enable(client, admin, graph.id).status_code == status.HTTP_200_OK


def test_s7_a_cycle_needs_max_node_executions(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b), ("c", c)], [("a", "b"), ("b", "c"), ("c", "b")])

    _rejected(_enable(client, admin, graph.id), "S7", "cycle", "max_node_executions")
    bounded = client.patch(
        f"/agent_graphs/{graph.id}",
        json={"enabled": True, "max_node_executions": 6},
        headers=admin,
    )
    assert bounded.status_code == status.HTTP_200_OK, bounded.text


def test_s7_max_node_executions_has_a_configured_maximum(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"])], max_node_executions=MAX_NODE_EXECUTIONS + 1)

    _rejected(_enable(client, admin, graph.id), "S7", str(MAX_NODE_EXECUTIONS))


def test_s8_a_gateway_is_only_an_entry_node(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
    gateway: Agent,
):
    graph = build([("a", agents["Alpha"]), ("gateway", gateway)], [("a", "gateway")])

    _rejected(_enable(client, admin, graph.id), "S8", "'Gateway'", "entry node")


def test_s8_a_gateway_enters_at_most_one_enabled_graph(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    gateway: Agent,
):
    first = build([("gateway", gateway)], name="First", enabled=True)
    second = build([("gateway", gateway)], name="Second")

    _rejected(_enable(client, admin, second.id), "S8", "'Gateway'", "'First'")
    assert first.id != second.id


def test_one_agent_may_back_several_nodes_with_different_keys(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    alpha = agents["Alpha"]
    graph = build([("first", alpha), ("again", alpha)], [("first", "again")])

    assert _enable(client, admin, graph.id).status_code == status.HTTP_200_OK


def test_a_node_key_is_unique_in_its_graph(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"])])

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "a", "agent_id": str(agents["Beta"].id)},
        headers=admin,
    )

    assert response.status_code == status.HTTP_409_CONFLICT
    assert "already has a node 'a'" in response.json()["detail"]["message"]


def test_a_node_needs_an_existing_agent(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    graph = build([("a", agents["Alpha"])])

    response = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "b", "agent_id": str(uuid.uuid4())},
        headers=admin,
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_a_graph_created_and_enabled_through_the_api_is_the_one_recoveries_run(
    client: TestClient,
    admin: Headers,
    agents: dict[str, Agent],
    scripted_agent: ScriptedAgent,
    make_agent_graph: MakeAgentGraph,
    make_robot_key: Callable[..., Any],
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    _ = make_agent_graph(Node("old", agents["Alpha"]), name="Old Default")
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
    graph_id = _create_graph(
        client, admin, name="New Default", is_default_recovery=True
    ).json()["id"]
    node = client.post(
        f"/agent_graphs/{graph_id}/nodes",
        json={
            "key": "worker",
            "agent_id": str(worker.id),
            "is_entry": True,
            "input_map": {"task": "recovery.task_name"},
        },
        headers=admin,
    )
    assert node.status_code == status.HTTP_201_CREATED, node.text
    assert _enable(client, admin, graph_id).status_code == status.HTTP_200_OK
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
    graphs = {g["name"]: g for g in client.get("/agent_graphs/", headers=admin).json()}
    assert graphs["New Default"]["is_default_recovery"] is True
    assert graphs["Old Default"]["is_default_recovery"] is False


def test_an_edge_condition_round_trips_and_a_malformed_one_is_rejected(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b), ("c", c)], [("a", "b", 0)])
    negated = {"not": {"field": "done", "op": "eq", "value": True}}

    added = _add_edge(client, admin, graph.id, graph.nodes, "a", "c", 1, negated)
    malformed = _add_edge(
        client, admin, graph.id, graph.nodes, "a", "c", 1, {"field": "done", "op": "is"}
    )
    changed = client.patch(
        f"/agent_graphs/{graph.id}/edges/{added.json()['id']}",
        json={"condition": None, "priority": 2},
        headers=admin,
    )

    assert added.status_code == status.HTTP_201_CREATED, added.text
    assert added.json()["condition"] == negated
    assert malformed.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert changed.status_code == status.HTTP_200_OK, changed.text
    assert changed.json()["condition"] is None
    assert changed.json()["priority"] == 2


def test_v5_changing_an_edge_must_keep_an_enabled_graph_valid(
    client: TestClient,
    admin: Headers,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, _ = agents.values()
    graph = build([("a", a), ("b", b)], [("a", "b")], enabled=True)
    edge_id = client.get(f"/agent_graphs/{graph.id}", headers=admin).json()["edges"][0][
        "id"
    ]

    response = client.patch(
        f"/agent_graphs/{graph.id}/edges/{edge_id}",
        json={"target_node_id": graph.nodes["a"]},
        headers=admin,
    )

    _rejected(response, "S1", "entry node 'a'", "incoming")


def test_v5_only_checks_the_agent_of_the_node_being_written(
    client: TestClient,
    admin: Headers,
    session: Session,
    build: Callable[..., Built],
    agents: dict[str, Agent],
):
    a, b, c = agents.values()
    graph = build([("a", a), ("b", b)], [("a", "b")], enabled=True)
    b.enabled = False
    session.add(b)
    session.commit()

    renamed = client.patch(
        f"/agent_graphs/{graph.id}", json={"description": "Renamed."}, headers=admin
    )
    added = client.post(
        f"/agent_graphs/{graph.id}/nodes",
        json={"key": "c", "agent_id": str(c.id)},
        headers=admin,
    )

    assert renamed.status_code == status.HTTP_200_OK, renamed.text
    _rejected(added, "S2", "'c'")
    assert "V5" not in added.text
