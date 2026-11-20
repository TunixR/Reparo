"""
Agent graph fixtures for unit tests.
"""

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers
from httpx2 import Response
from pydantic import BaseModel
from sqlalchemy import Engine
from sqlmodel import Session, col, select

import database.populators.agents as agents_populator
from agent_tools.utils import SET_CONTINUATION_FN_MODULE
from database.agent_graphs.models import AgentGraph, AgentGraphEdge, AgentGraphNode
from database.agents.models import Agent, AgentTool, AgentType, Argument
from database.populators.agents import populate_agents
from database.populators.tools import populate_tools
from database.provider.models import Router
from database.tools.models import Tool


class Finding(BaseModel):
    reason: str
    tags: list[str]


class GraphReport(BaseModel):
    """Output model of ``graph_agents``."""

    done: bool
    success: bool
    attempts: int
    note: str | None
    finding: Finding
    extra: dict[str, int]


GRAPH_REPORT = f"{__name__}.GraphReport"


def argument(name: str) -> Argument:
    """A string argument called ``name``."""
    return Argument(name=name, description=name, type="str", json_type="string")


@dataclass(frozen=True)
class Node:
    """A node to create: its key, its agent and its input map."""

    key: str
    agent: Agent
    input_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """An edge to create; without a priority, its place in ``edges`` is its priority."""

    source: str
    target: str
    condition: dict[str, Any] | None = None
    priority: int | None = None


type MakeAgentGraph = Callable[..., AgentGraph]
"""Factory returned by ``make_agent_graph``: nodes and edges in, the persisted graph out."""


@pytest.fixture
def make_agent_graph(session: Session) -> MakeAgentGraph:
    """Return a factory that persists an agent graph.

    The first node is the entry node. ``edges`` are `Edge`s or unconditional
    ``(source, target)`` key pairs; without them, the nodes are chained in order. A
    new default recovery graph takes over from the previous one.
    """

    def _make_agent_graph(
        *nodes: Node,
        edges: Sequence[Edge | tuple[str, str]] | None = None,
        name: str | None = None,
        enabled: bool = True,
        is_default_recovery: bool = True,
        max_node_executions: int | None = None,
        execution_timeout_s: float | None = None,
        node_timeout_s: float | None = None,
    ) -> AgentGraph:
        if is_default_recovery:
            for previous in session.exec(
                select(AgentGraph).where(col(AgentGraph.is_default_recovery).is_(True))
            ):
                previous.is_default_recovery = False
                session.add(previous)
            session.commit()
        graph = AgentGraph(
            name=name or f"Test Graph {uuid.uuid4()}",
            description="A graph for testing.",
            enabled=enabled,
            is_default_recovery=is_default_recovery,
            max_node_executions=max_node_executions,
            execution_timeout_s=execution_timeout_s,
            node_timeout_s=node_timeout_s,
        )
        rows = {
            node.key: AgentGraphNode(
                agent_graph_id=graph.id,
                key=node.key,
                agent_id=node.agent.id,
                is_entry=index == 0,
                input_map=dict(node.input_map),
            )
            for index, node in enumerate(nodes)
        }
        graph.nodes = list(rows.values())
        if edges is None:
            edges = [(a.key, b.key) for a, b in zip(nodes, nodes[1:], strict=False)]
        normalized = [edge if isinstance(edge, Edge) else Edge(*edge) for edge in edges]
        graph.edges = [
            AgentGraphEdge.model_validate(
                {
                    "agent_graph_id": graph.id,
                    "source_node_id": rows[edge.source].id,
                    "target_node_id": rows[edge.target].id,
                    "priority": index if edge.priority is None else edge.priority,
                    "condition": edge.condition,
                }
            )
            for index, edge in enumerate(normalized)
        ]
        session.add(graph)
        session.commit()
        session.refresh(graph)
        return graph

    return _make_agent_graph


def create_graph_via_api(
    client: TestClient, headers: Headers, **fields: Any
) -> Response:
    """POST a graph with a unique name; ``fields`` override the body."""
    body = {"name": f"Graph {uuid.uuid4()}", "description": "A test graph."} | fields
    return client.post("/agent_graphs/", json=body, headers=headers)


def add_edge_via_api(
    client: TestClient,
    headers: Headers,
    graph_id: str,
    node_ids: Mapping[str, str],
    source: str,
    target: str,
    priority: int = 0,
    condition: dict[str, Any] | None = None,
) -> Response:
    """POST an edge between the nodes with keys ``source`` and ``target``."""
    return client.post(
        f"/agent_graphs/{graph_id}/edges",
        json={
            "source_node_id": node_ids[source],
            "target_node_id": node_ids[target],
            "priority": priority,
            "condition": condition,
        },
        headers=headers,
    )


def enable_via_api(client: TestClient, headers: Headers, graph_id: str) -> Response:
    return client.patch(
        f"/agent_graphs/{graph_id}", json={"enabled": True}, headers=headers
    )


type NodeSpec = (
    tuple[str, Agent]
    | tuple[str, Agent, bool]
    | tuple[str, Agent, bool, Mapping[str, str]]
)
"""``(key, agent[, is_entry[, input_map]])``."""

type EdgeSpec = (
    tuple[str, str] | tuple[str, str, int] | tuple[str, str, int, dict[str, Any]]
)
"""``(source, target[, priority[, condition]])``, by node key."""


@dataclass(frozen=True)
class ApiGraph:
    """A graph made through the API: its id and its node ids by key."""

    id: str
    nodes: Mapping[str, str]


type BuildAgentGraph = Callable[..., ApiGraph]
"""Factory returned by ``build_agent_graph``: node and edge specs in, the graph out."""


@pytest.fixture
def build_agent_graph(client: TestClient, admin_headers: Headers) -> BuildAgentGraph:
    """Return a factory that builds a graph through the API as an administrator.

    The first node is the entry unless a spec says otherwise. Without a priority,
    an edge's place in the list is its priority. With ``enabled``, the graph is
    enabled once complete. Other keyword arguments are graph fields.
    """

    def _build(
        nodes: Sequence[NodeSpec],
        edges: Sequence[EdgeSpec] = (),
        enabled: bool = False,
        **fields: Any,
    ) -> ApiGraph:
        created = create_graph_via_api(client, admin_headers, **fields)
        assert created.status_code == status.HTTP_201_CREATED, created.text
        graph_id: str = created.json()["id"]
        ids: dict[str, str] = {}
        for index, spec in enumerate(nodes):
            input_map: Mapping[str, str] = {}
            match spec:
                case (key, agent):
                    is_entry = index == 0
                case (key, agent, is_entry):
                    pass
                case (key, agent, is_entry, input_map):
                    pass
            response = client.post(
                f"/agent_graphs/{graph_id}/nodes",
                json={
                    "key": key,
                    "agent_id": str(agent.id),
                    "is_entry": is_entry,
                    "input_map": dict(input_map),
                },
                headers=admin_headers,
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
            response = add_edge_via_api(
                client,
                admin_headers,
                graph_id,
                ids,
                source,
                target,
                priority,
                condition,
            )
            assert response.status_code == status.HTTP_201_CREATED, response.text
        if enabled:
            response = enable_via_api(client, admin_headers, graph_id)
            assert response.status_code == status.HTTP_200_OK, response.text
        return ApiGraph(graph_id, ids)

    return _build


@pytest.fixture
def graph_agents(
    make_agent: Callable[..., Agent], mock_router: Router
) -> dict[str, Agent]:
    """Three enabled agents, by name, to back graph nodes; each outputs a `GraphReport`."""
    return {
        name: make_agent(router=mock_router, name=name, response_model=GRAPH_REPORT)
        for name in ("Alpha", "Beta", "Gamma")
    }


@pytest.fixture
def gui_agent(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    """A GUI agent taking a ``task``; it outputs a `GuiRunReport`."""
    return make_agent(
        router=mock_router,
        name="Clicker",
        type=AgentType.GuiAgent,
        arguments=[argument("task")],
    )


@pytest.fixture
def disabled_agent(
    session: Session, make_agent: Callable[..., Agent], mock_router: Router
) -> Agent:
    agent = make_agent(router=mock_router, name="Dormant")
    agent.enabled = False
    session.add(agent)
    session.commit()
    return agent


@pytest.fixture
def tasked_agent(make_agent: Callable[..., Agent], mock_router: Router) -> Agent:
    """An agent taking a ``task``; it outputs a `GraphReport`."""
    return make_agent(
        router=mock_router,
        name="Tasked",
        arguments=[argument("task")],
        response_model=GRAPH_REPORT,
    )


@pytest.fixture
def continuation_agent(
    session: Session, make_agent: Callable[..., Agent], mock_router: Router
) -> Agent:
    """An agent with the set_continuation tool; it outputs a `GraphReport`."""
    agent = make_agent(
        router=mock_router, name="Continuer", response_model=GRAPH_REPORT
    )
    tool = Tool(
        name="set_continuation",
        description="Set where the robot continues.",
        fn_module=SET_CONTINUATION_FN_MODULE,
    )
    session.add(tool)
    session.commit()
    session.add(
        AgentTool(agent_id=agent.id, tool_id=tool.id, limit=None, required=True)
    )
    session.commit()
    session.refresh(agent)
    return agent


@pytest.fixture
def seeded_agents(
    engine: Engine,
    session: Session,
    make_router: Callable[..., Router],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Agent]:
    """Seed the tools and agents of ``agents.json``, every role on one vision router.

    Returns the seeded agents by name.
    """
    router = make_router(
        capabilities=[Router.Capability.TEXT, Router.Capability.VISION],
        vision_backbone=Router.VisionBackbone.QWEN3_5,
    )
    monkeypatch.setattr(
        agents_populator,
        "find_router",
        lambda populator_session, _role: populator_session.get(Router, router.id),  # pyright: ignore[reportUnknownLambdaType]
    )
    populate_tools(engine)
    populate_agents(engine)
    session.expire_all()
    return {agent.name: agent for agent in session.exec(select(Agent)).unique()}
