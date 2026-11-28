"""Paging of the history lists: traces and robot exceptions.

Each list is tried against the rows a recovery leaves: an agent trace carries GUI
traces and a sub-trace, so the lists that eager-load them by join fetch several SQL
rows per entity. A page must still hold ``limit`` entities.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers
from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel

from database.agents.models import Agent
from database.keys.models import RobotKey
from database.logging.models import (
    AgentGraphTrace,
    AgentTrace,
    GUITrace,
    RobotException,
    SubAgentTrace,
    ToolTrace,
)
from database.tools.models import Tool

START = datetime(2026, 1, 1, tzinfo=UTC)
GUI_STEPS_PER_TRACE = 3


@dataclass
class _Db:
    """The factories that fill the lists, and the agent traces of each recovery."""

    session: Session
    agent: Agent
    tool: Tool
    make_robot_exception: Callable[..., RobotException]
    make_robot_key: Callable[..., RobotKey]
    make_agent_trace: Callable[..., AgentTrace]
    make_gui_trace: Callable[..., GUITrace]
    make_tool_trace: Callable[..., ToolTrace]
    traces: dict[UUID, list[AgentTrace]] = field(default_factory=dict)

    def exception(self) -> UUID:
        return self.make_robot_exception().id

    def key(self) -> UUID:
        return self.make_robot_key().id

    def agent_trace(
        self, exception_id: UUID, at: datetime, gui_steps: int = GUI_STEPS_PER_TRACE
    ) -> AgentTrace:
        """An agent trace with GUI steps, and the recovery's previous one as sub-trace.

        A trace has at most one parent, so the recovery's traces form a chain.
        """
        trace = self.make_agent_trace(
            agent=self.agent, robot_exception_id=exception_id, created_at=at
        )
        for _ in range(gui_steps):
            _ = self.make_gui_trace(agent_trace=trace, created_at=at)
        earlier = self.traces.setdefault(exception_id, [])
        if earlier:
            self.session.add(
                SubAgentTrace(parent_trace_id=trace.id, child_trace_id=earlier[-1].id)
            )
        self.session.commit()
        earlier.append(trace)
        return trace

    def first_agent_trace(self, exception_id: UUID, gui_steps: int) -> AgentTrace:
        """The agent trace the GUI or tool steps of a recovery hang from."""
        if exception_id not in self.traces:
            _ = self.agent_trace(exception_id, START, gui_steps)
        return self.traces[exception_id][0]


type _Row = Callable[[_Db, UUID | None, datetime], UUID]


@dataclass(frozen=True)
class _List:
    """A paginated list: where it is, what it is filtered by, how to add a row."""

    path: str
    model: type[SQLModel]
    scope: Callable[[_Db], UUID | None]
    row: _Row
    filter_param: str | None = None

    @property
    def name(self) -> str:
        return self.path + (f"?{self.filter_param}" if self.filter_param else "")

    @property
    def filtered(self) -> bool:
        return self.filter_param is not None or "{scope}" in self.path

    def request(self, scope: UUID | None) -> tuple[str, dict[str, str]]:
        params = {self.filter_param: str(scope)} if self.filter_param else {}
        return f"/logging/{self.path.format(scope=scope)}", params


def _agent_trace(db: _Db, scope: UUID | None, at: datetime) -> UUID:
    assert scope is not None
    return db.agent_trace(scope, at).id


def _gui_step(db: _Db, scope: UUID | None, at: datetime) -> UUID:
    assert scope is not None
    trace = db.first_agent_trace(scope, gui_steps=0)
    return db.make_gui_trace(agent_trace=trace, created_at=at).id


def _tool_step(db: _Db, scope: UUID | None, at: datetime) -> UUID:
    assert scope is not None
    trace = db.first_agent_trace(scope, gui_steps=GUI_STEPS_PER_TRACE)
    return db.make_tool_trace(agent_trace=trace, tool=db.tool, created_at=at).id


def _graph_run(db: _Db, scope: UUID | None, at: datetime) -> UUID:
    assert scope is not None
    run = AgentGraphTrace(robot_exception_id=scope, created_at=at)
    db.session.add(run)
    db.session.commit()
    return run.id


def _robot_exception(db: _Db, scope: UUID | None, at: datetime) -> UUID:
    return db.make_robot_exception(robot_key_id=scope, created_at=at).id


def _recovery_lists(path: str, model: type[SQLModel], row: _Row) -> list[_List]:
    """A list of every recovery's rows, and the same list of one recovery's."""
    return [
        _List(path, model, _Db.exception, row),
        _List(path, model, _Db.exception, row, filter_param="robot_exception_id"),
    ]


LISTS = [
    *_recovery_lists("agent_traces", AgentTrace, _agent_trace),
    *_recovery_lists("gui_traces", GUITrace, _gui_step),
    *_recovery_lists("tool_traces", ToolTrace, _tool_step),
    *_recovery_lists("agent_graph_traces", AgentGraphTrace, _graph_run),
    _List("robot_exceptions", RobotException, lambda _db: None, _robot_exception),
    _List("key/{scope}", RobotException, _Db.key, _robot_exception),
]
FILTERED_LISTS = [lst for lst in LISTS if lst.filtered]


def _name(lst: _List) -> str:
    return lst.name


@pytest.fixture
def db(
    session: Session,
    mock_agent: Agent,
    mock_tool: Tool,
    make_robot_exception: Callable[..., RobotException],
    make_robot_key: Callable[..., RobotKey],
    make_agent_trace: Callable[..., AgentTrace],
    make_gui_trace: Callable[..., GUITrace],
    make_tool_trace: Callable[..., ToolTrace],
) -> _Db:
    return _Db(
        session,
        mock_agent,
        mock_tool,
        make_robot_exception,
        make_robot_key,
        make_agent_trace,
        make_gui_trace,
        make_tool_trace,
    )


def _seed(db: _Db, lst: _List, count: int) -> tuple[UUID | None, list[str]]:
    """A new scope with ``count`` rows, two to a timestamp; their ids newest first.

    Rows created at the same time come highest id first: the tiebreaker.
    """
    scope = lst.scope(db)
    stamps = [START + timedelta(seconds=i // 2) for i in range(count)]
    rows = sorted(((at, lst.row(db, scope, at)) for at in stamps), reverse=True)
    return scope, [str(row_id) for _, row_id in rows]


def _response(
    client: TestClient, headers: Headers, lst: _List, scope: UUID | None, **page: int
) -> tuple[int, list[dict[str, object]]]:
    path, params = lst.request(scope)
    response = client.get(path, params=params | page, headers=headers)
    return response.status_code, response.json()


def _get(
    client: TestClient, headers: Headers, lst: _List, scope: UUID | None, **page: int
) -> list[str]:
    code, body = _response(client, headers, lst, scope, **page)
    assert code == status.HTTP_200_OK, body
    return [str(item["id"]) for item in body]


@pytest.mark.parametrize("lst", LISTS, ids=_name)
def test_list_without_limit_returns_the_first_100_newest_first(
    lst: _List, db: _Db, user_headers: Headers, client: TestClient
):
    scope, newest_first = _seed(db, lst, 101)

    assert _get(client, user_headers, lst, scope) == newest_first[:100]


@pytest.mark.parametrize("page", [{"limit": 1001}, {"limit": 0}, {"offset": -1}])
@pytest.mark.parametrize("lst", LISTS, ids=_name)
def test_list_refuses_a_page_out_of_bounds(
    lst: _List,
    page: dict[str, int],
    db: _Db,
    user_headers: Headers,
    client: TestClient,
):
    code, _ = _response(client, user_headers, lst, lst.scope(db), **page)

    assert code == status.HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize("lst", LISTS, ids=_name)
def test_list_pages_hold_every_row_once_in_order(
    lst: _List, db: _Db, user_headers: Headers, client: TestClient
):
    scope, newest_first = _seed(db, lst, 7)

    pages = [
        _get(client, user_headers, lst, scope, limit=3, offset=offset)
        for offset in (0, 3, 6, 9)
    ]

    assert pages == [newest_first[:3], newest_first[3:6], newest_first[6:], []]
    assert _get(client, user_headers, lst, scope, limit=1000) == newest_first


@pytest.mark.parametrize("lst", FILTERED_LISTS, ids=_name)
def test_filtered_list_pages_only_the_rows_of_its_scope(
    lst: _List, db: _Db, user_headers: Headers, client: TestClient
):
    ours, newest_first = _seed(db, lst, 5)
    _ = _seed(db, lst, 5)

    assert _get(client, user_headers, lst, ours, limit=2, offset=1) == newest_first[1:3]


@pytest.mark.parametrize("lst", LISTS, ids=_name)
def test_list_pages_in_sql(
    lst: _List, db: _Db, user_headers: Headers, client: TestClient, engine: Engine
):
    scope, _ = _seed(db, lst, 3)
    statements: list[str] = []

    def record(_conn: object, _cursor: object, statement: str, *_: object) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        _ = _get(client, user_headers, lst, scope, limit=2, offset=1)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    table = f"FROM {lst.model.__tablename__}"
    assert any(
        table in statement and "LIMIT ? OFFSET ?" in statement
        for statement in statements
    )
