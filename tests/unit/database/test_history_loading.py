"""Loading a row must not load the history that grows with every recovery or login.

Each test persists some history, loads the row in a fresh session (so the
identity map can't answer for the database) and inspects the SQL it emitted.
"""

import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import Headers
from sqlalchemy import Engine, event
from sqlalchemy.exc import InvalidRequestError
from sqlmodel import Session

import database.engine
from agent_tools.hooks import AgentLoggingHook
from database.agents.models import Agent, AgentTool
from database.auth.models import User
from database.logging.models import AgentTrace, GUITrace, ToolTrace
from database.tools.models import Tool
from middlewares.auth import get_current_user
from tests.unit.agent_tools._shared.fakes import FakeBeforeInvocationEvent
from tests.unit.shared.auth_helpers import (
    make_access_token,
    make_auth_headers,
    persist_user_session,
)


@contextmanager
def emitted_sql(engine: Engine) -> Iterator[list[str]]:
    """The statements ``engine`` executes inside the block, lowercased."""
    statements: list[str] = []

    def _record(_conn: object, _cursor: object, statement: str, *_: object) -> None:
        statements.append(statement.lower())

    event.listen(engine, "before_cursor_execute", _record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", _record)


def tables_read(statements: list[str]) -> set[str]:
    """The tables named after FROM or JOIN in ``statements``."""
    return {
        table
        for statement in statements
        for table in re.findall(r"(?:from|join) (\w+)", statement)
    }


def reads_of(table: str, statements: list[str]) -> list[str]:
    """The statements in ``statements`` that read ``table``."""
    return [s for s in statements if table in tables_read([s])]


@pytest.mark.parametrize(
    ("model", "fixture", "history"),
    [
        (Agent, "mock_agent", "traces"),
        (Tool, "mock_tool", "traces"),
        (User, "mock_user", "sessions"),
        (AgentTrace, "mock_agent_trace", "gui_traces"),
    ],
)
def test_reading_history_through_a_relationship_raises(
    model: type[Agent | Tool | User | AgentTrace],
    fixture: str,
    history: str,
    engine: Engine,
    request: pytest.FixtureRequest,
):
    """History is read with a bounded query, never by walking the relationship."""
    row_id = request.getfixturevalue(fixture).id
    with Session(engine) as fresh:
        row = fresh.get_one(model, row_id)
        with pytest.raises(InvalidRequestError, match="lazy='raise'"):
            _ = getattr(row, history)


def test_loading_an_agent_reads_none_of_its_or_its_tools_traces(
    engine: Engine,
    session: Session,
    mock_agent: Agent,
    mock_tool: Tool,
    make_agent_trace: Callable[..., AgentTrace],
    make_tool_trace: Callable[..., ToolTrace],
):
    session.add(
        AgentTool(
            agent_id=mock_agent.id, tool_id=mock_tool.id, limit=None, required=False
        )
    )
    session.commit()
    trace = make_agent_trace(agent=mock_agent)
    _ = make_tool_trace(agent_trace=trace, tool=mock_tool)

    with Session(engine) as fresh, emitted_sql(engine) as statements:
        agent = fresh.get(Agent, mock_agent.id)

    assert agent is not None
    assert [link.tool.name for link in agent.tools] == [mock_tool.name]
    assert tables_read(statements).isdisjoint({"agenttrace", "tooltrace"})


def test_authenticating_is_one_statement_for_the_session_and_its_user(
    engine: Engine, session: Session, mock_user: User
):
    for _ in range(3):
        _ = persist_user_session(session, mock_user)
    token = make_access_token(session, mock_user)

    with Session(engine) as fresh, emitted_sql(engine) as statements:
        user = get_current_user(session=fresh, token=token)

    assert user.id == mock_user.id
    assert len(statements) == 1


def test_logging_in_reads_none_of_the_users_past_sessions(
    engine: Engine, session: Session, mock_user: User, client: TestClient
):
    for _ in range(3):
        _ = persist_user_session(session, mock_user)

    with emitted_sql(engine) as statements:
        response = client.post(
            "/auth/login",
            data={"username": mock_user.username, "password": "password123"},
        )

    assert response.status_code == status.HTTP_200_OK
    # Only the new session's own row is read back, by its id.
    assert len(reads_of("usersession", statements)) == 1


@pytest.mark.parametrize("path", ["/auth/users", "/auth/users/search"])
def test_listing_users_reads_none_of_their_sessions(
    path: str,
    engine: Engine,
    session: Session,
    mock_user: User,
    admin_headers: Headers,
    client: TestClient,
):
    for _ in range(3):
        _ = persist_user_session(session, mock_user)

    with emitted_sql(engine) as statements:
        response = client.get(path, headers=admin_headers)

    assert response.status_code == status.HTTP_200_OK
    assert mock_user.username in {user["username"] for user in response.json()}
    # Only authenticating the admin reads a session: their own, by its id.
    assert len(reads_of("usersession", statements)) == 1


def _start_sub_agent(parent: AgentLoggingHook) -> None:
    AgentLoggingHook(
        agent_id=parent.agent_id,
        invocation_state={},
        parent_trace_id=parent.agent_trace_id,
    ).log_start(FakeBeforeInvocationEvent())


def test_starting_a_sub_agent_reads_no_agent_trace(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, mock_agent: Agent
):
    monkeypatch.setattr(database.engine, "general_engine", engine)
    hook = AgentLoggingHook(agent_id=mock_agent.id, invocation_state={})
    hook.log_start(FakeBeforeInvocationEvent())
    _start_sub_agent(hook)

    with emitted_sql(engine) as statements:
        _start_sub_agent(hook)

    assert "subagenttrace" in tables_read(statements)
    assert "agenttrace" not in tables_read(statements)


def test_updating_a_running_agent_trace_reads_none_of_its_gui_or_sub_agent_traces(
    engine: Engine,
    session: Session,
    monkeypatch: pytest.MonkeyPatch,
    mock_agent: Agent,
    make_gui_trace: Callable[..., GUITrace],
):
    monkeypatch.setattr(database.engine, "general_engine", engine)
    hook = AgentLoggingHook(agent_id=mock_agent.id, invocation_state={})
    hook.log_start(FakeBeforeInvocationEvent())
    _start_sub_agent(hook)
    trace = session.get_one(AgentTrace, hook.agent_trace_id)
    for _ in range(3):
        _ = make_gui_trace(agent_trace=trace)

    with emitted_sql(engine) as statements:
        hook.update_trace(finished=True)

    assert session.get_one(
        AgentTrace, hook.agent_trace_id, populate_existing=True
    ).finished_at
    assert tables_read(statements).isdisjoint({"guitrace", "subagenttrace"})


@pytest.mark.parametrize("path", ["/logging/gui_traces", "/logging/tool_traces"])
def test_listing_a_runs_steps_reads_none_of_their_agent_traces(
    path: str,
    engine: Engine,
    session: Session,
    client: TestClient,
    mock_user: User,
    mock_agent_trace: AgentTrace,
    mock_tool: Tool,
    make_gui_trace: Callable[..., GUITrace],
    make_tool_trace: Callable[..., ToolTrace],
):
    headers = make_auth_headers(mock_user, session)
    _ = make_gui_trace(agent_trace=mock_agent_trace)
    _ = make_tool_trace(agent_trace=mock_agent_trace, tool=mock_tool)

    with emitted_sql(engine) as statements:
        response = client.get(path, headers=headers)

    assert len(response.json()) == 1
    assert "agenttrace" not in tables_read(statements)
