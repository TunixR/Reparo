"""Agents used by agent graphs can't be changed in ways that break those graphs."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers

from database.agent_graphs.models import AgentGraph
from database.agents.models import Agent, AgentType
from tests.unit.fixtures.agent_graph_fixtures import (
    Edge,
    MakeAgentGraph,
    Node,
)

FINDING = "tests.unit.fixtures.agent_graph_fixtures.Finding"
DONE = {"field": "done", "op": "eq", "value": True}


def _listed(graphs: list[dict[str, str]]) -> list[str]:
    return sorted(graph["name"] for graph in graphs)


def test_disabling_an_agent_of_an_enabled_graph_is_a_conflict_listing_the_graphs(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
):
    alpha, beta = graph_agents["Alpha"], graph_agents["Beta"]
    for name in ("First", "Second"):
        _ = make_agent_graph(
            Node("a", alpha), Node("b", beta), name=name, is_default_recovery=False
        )
    _ = make_agent_graph(
        Node("a", alpha), name="Draft", enabled=False, is_default_recovery=False
    )

    response = client.patch(
        f"/agents/{alpha.id}", json={"enabled": False}, headers=admin_headers
    )

    assert response.status_code == status.HTTP_409_CONFLICT, response.text
    assert _listed(response.json()["detail"]["graphs"]) == ["First", "Second"]
    assert client.get(f"/agents/{alpha.id}", headers=admin_headers).json()["enabled"]


def test_disabling_an_agent_only_drafts_use_is_allowed(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
):
    alpha = graph_agents["Alpha"]
    _ = make_agent_graph(Node("a", alpha), enabled=False, is_default_recovery=False)

    response = client.patch(
        f"/agents/{alpha.id}", json={"enabled": False}, headers=admin_headers
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["enabled"] is False


def test_enabling_an_agent_used_by_graphs_is_always_allowed(
    client: TestClient,
    admin_headers: Headers,
    disabled_agent: Agent,
    make_agent_graph: MakeAgentGraph,
):
    _ = make_agent_graph(
        Node("a", disabled_agent), enabled=False, is_default_recovery=False
    )

    response = client.patch(
        f"/agents/{disabled_agent.id}", json={"enabled": True}, headers=admin_headers
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["enabled"] is True


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "draft"])
def test_deleting_an_agent_any_graph_uses_is_a_conflict_listing_the_graphs(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
    enabled: bool,
):
    alpha = graph_agents["Alpha"]
    _ = make_agent_graph(
        Node("a", alpha), name="Pipeline", enabled=enabled, is_default_recovery=False
    )

    response = client.delete(f"/agents/{alpha.id}", headers=admin_headers)

    assert response.status_code == status.HTTP_409_CONFLICT, response.text
    assert _listed(response.json()["detail"]["graphs"]) == ["Pipeline"]
    found = client.get(f"/agents/{alpha.id}", headers=admin_headers)
    assert found.status_code == status.HTTP_200_OK


def test_deleting_an_agent_no_graph_uses_still_succeeds(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    make_agent_graph: MakeAgentGraph,
):
    _ = make_agent_graph(Node("a", graph_agents["Alpha"]), is_default_recovery=False)
    beta = graph_agents["Beta"]

    response = client.delete(f"/agents/{beta.id}", headers=admin_headers)

    assert response.status_code == status.HTTP_204_NO_CONTENT, response.text


@pytest.fixture
def routed_by_alpha(
    graph_agents: dict[str, Agent], make_agent_graph: MakeAgentGraph
) -> MakeAgentGraph:
    """Return a factory for graphs Alpha -> Beta, past Alpha only when it's done."""

    def _routed(**settings: Any) -> AgentGraph:
        return make_agent_graph(
            Node("alpha", graph_agents["Alpha"]),
            Node("beta", graph_agents["Beta"]),
            edges=[Edge("alpha", "beta", DONE)],
            **{"is_default_recovery": False} | settings,
        )

    return _routed


def test_a_response_model_change_that_breaks_a_condition_is_a_conflict(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    routed_by_alpha: MakeAgentGraph,
):
    _ = routed_by_alpha(name="Routed")
    _ = routed_by_alpha(name="Draft", enabled=False)
    alpha = graph_agents["Alpha"]

    response = client.patch(
        f"/agents/{alpha.id}", json={"response_model": FINDING}, headers=admin_headers
    )

    assert response.status_code == status.HTTP_409_CONFLICT, response.text
    detail = response.json()["detail"]
    assert _listed(detail["graphs"]) == ["Routed"]
    [violation] = detail["violations"]
    assert violation.startswith("Routed: The condition on the edge 'alpha' → 'beta'")
    assert "'done'" in violation
    got = client.get(f"/agents/{alpha.id}", headers=admin_headers).json()
    assert got["response_model"] == graph_agents["Alpha"].response_model


def test_a_type_change_that_puts_a_gateway_past_the_entry_is_a_conflict(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    routed_by_alpha: MakeAgentGraph,
):
    _ = routed_by_alpha(name="Routed")
    beta = graph_agents["Beta"]

    response = client.patch(
        f"/agents/{beta.id}",
        json={"type": AgentType.GatewayAgent.value},
        headers=admin_headers,
    )

    assert response.status_code == status.HTTP_409_CONFLICT, response.text
    assert _listed(response.json()["detail"]["graphs"]) == ["Routed"]


def test_changes_that_keep_enabled_graphs_valid_still_succeed(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    routed_by_alpha: MakeAgentGraph,
):
    _ = routed_by_alpha(name="Routed")
    beta = graph_agents["Beta"]

    response = client.patch(
        f"/agents/{beta.id}",
        json={"response_model": FINDING, "prompt": "Be brief."},
        headers=admin_headers,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.json()["response_model"] == FINDING


def test_violations_an_enabled_graph_already_had_dont_block_an_agent_change(
    client: TestClient,
    admin_headers: Headers,
    graph_agents: dict[str, Agent],
    routed_by_alpha: MakeAgentGraph,
):
    _ = routed_by_alpha(name="Unfit default", is_default_recovery=True)
    beta = graph_agents["Beta"]

    response = client.patch(
        f"/agents/{beta.id}", json={"response_model": FINDING}, headers=admin_headers
    )

    assert response.status_code == status.HTTP_200_OK, response.text
