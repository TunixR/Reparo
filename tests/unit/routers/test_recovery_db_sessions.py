"""A recovery holds no database connection while it waits on the robot or the agents.

Each WebSocket lives for minutes; a connection checked out for all of it would let a
handful of concurrent recoveries exhaust the pool for every HTTP request.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy import event
from sqlalchemy.orm import SessionTransaction
from sqlmodel import Session, select

import routers.evals as evals_module
from database.agent_graphs.models import AgentGraph
from database.agent_graphs.runner import GraphOutcome, NodeRun
from database.agents.behaviours import AgentBehaviour
from database.agents.models import Agent
from database.logging.models import RobotException
from database.provider.models import Router
from robot.channel import RobotConnection
from tests.unit.fixtures.agent_graph_fixtures import MakeAgentGraph, Node
from tests.unit.fixtures.robot_ws_fixtures import receive_skipping_pings

KEY_RAW = "test-robot-key-db-sessions-0001"


class _Report(BaseModel):
    """A node output without ``success``: the recovery ends not recovered."""

    result: str = "ran"


class ConnectionHolders:
    """The app's sessions that hold a connection, in a transaction, right now."""

    def __init__(self, fixture_session: Session) -> None:
        self._ignored = fixture_session
        self._holding: set[Session] = set()

    def began(self, session: Session, *_: Any) -> None:
        if session is not self._ignored:
            self._holding.add(session)

    def ended(self, session: Session, transaction: SessionTransaction) -> None:
        if transaction.parent is None:
            self._holding.discard(session)

    def now(self) -> set[Session]:
        return {s for s in self._holding if s.in_transaction()}


@pytest.fixture
def holders(session: Session) -> Iterator[ConnectionHolders]:
    tracked = ConnectionHolders(session)
    event.listen(Session, "after_begin", tracked.began)
    event.listen(Session, "after_transaction_end", tracked.ended)
    yield tracked
    event.remove(Session, "after_begin", tracked.began)
    event.remove(Session, "after_transaction_end", tracked.ended)


@pytest.fixture
def robot_key(make_robot_key) -> None:  # pyright: ignore[reportMissingParameterType]
    _ = make_robot_key(name="DB sessions", enabled=True, key_raw=KEY_RAW)


@pytest.fixture
def graph(
    robot_key: None,  # pyright: ignore[reportUnusedParameter]
    make_agent: Callable[..., Agent],
    mock_router: Router,
    make_agent_graph: MakeAgentGraph,
) -> AgentGraph:
    gateway = make_agent(router=mock_router, name="Gateway")
    return make_agent_graph(Node("gateway", gateway))


@pytest.fixture
def node_holders(
    holders: ConnectionHolders, monkeypatch: pytest.MonkeyPatch
) -> list[set[Session]]:
    """The connection holders each node run saw; the nodes just report they ran."""
    seen: list[set[Session]] = []

    async def _node(
        self: AgentBehaviour, _state: dict[str, Any], _input_map: Mapping[str, str]
    ) -> NodeRun:
        seen.append(holders.now())
        # What the agents read off the graph's ORM objects is still loaded.
        assert self.agent.router.model_name
        return NodeRun(output=_Report())

    monkeypatch.setattr(AgentBehaviour, "run_node", _node)
    return seen


def _recover(client: TestClient, payload: dict[str, Any]) -> dict[str, Any]:
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": KEY_RAW}
    ) as ws:
        ws.send_json(payload)
        return receive_skipping_pings(ws)


def test_no_connection_is_held_while_waiting_for_the_payload(
    client: TestClient,
    holders: ConnectionHolders,
    graph: AgentGraph,  # pyright: ignore[reportUnusedParameter]
    node_holders: list[set[Session]],  # pyright: ignore[reportUnusedParameter]
    valid_recovery_payload: dict[str, Any],
):
    with client.websocket_connect(
        "/recovery/robot_exception/ws", headers={"X-ROBOT-KEY": KEY_RAW}
    ) as ws:
        # Accepted: the robot key was checked, and the payload is awaited.
        assert holders.now() == set()
        ws.send_json(valid_recovery_payload)
        assert receive_skipping_pings(ws)["type"] == "done"


def test_no_connection_is_held_while_the_graph_runs(
    client: TestClient,
    holders: ConnectionHolders,
    graph: AgentGraph,
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[set[Session]] = []
    names: list[str] = []

    async def _run(self: AgentGraph, state: dict[str, Any]) -> GraphOutcome:
        seen.append(holders.now())
        names.append(self.name)
        assert isinstance(state["websocket"], RobotConnection)
        return GraphOutcome("gateway", None, 1)

    monkeypatch.setattr(AgentGraph, "run", _run)

    message = _recover(client, valid_recovery_payload)

    assert message["type"] == "done"
    assert seen == [set()]
    assert names == [graph.name]


def test_no_connection_is_held_while_a_node_runs(
    client: TestClient,
    session: Session,
    graph: AgentGraph,  # pyright: ignore[reportUnusedParameter]
    node_holders: list[set[Session]],
    valid_recovery_payload: dict[str, Any],
):
    message = _recover(client, valid_recovery_payload)

    assert message["type"] == "done"
    assert node_holders == [set()]
    session.expire_all()
    exception = session.exec(select(RobotException)).one()
    assert exception.infered_success is False
    assert exception.recovery_context.task_name == valid_recovery_payload["task_name"]


def test_no_connection_is_held_while_an_eval_runs(
    client: TestClient,
    session: Session,
    holders: ConnectionHolders,
    robot_key: None,  # pyright: ignore[reportUnusedParameter]
    eval_gui_agent: Agent,
    valid_recovery_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[set[Session]] = []

    async def _recovery(*_: Any) -> dict[str, Any]:
        seen.append(holders.now())
        return {"success": True}

    monkeypatch.setattr(evals_module, "_run_standalone_recovery", _recovery)

    with client.websocket_connect(
        "/evals/recovery/ws",
        headers={"X-ROBOT-KEY": KEY_RAW, "X-AGENT-ID": str(eval_gui_agent.id)},
    ) as ws:
        assert holders.now() == set()
        ws.send_json(valid_recovery_payload | {"recovery_task": "Approve it"})
        message = receive_skipping_pings(ws)

    assert message["type"] == "eval_result"
    assert seen == [set()]
    session.expire_all()
    assert session.exec(select(RobotException)).one().infered_success is True
