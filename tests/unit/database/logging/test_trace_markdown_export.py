"""The markdown exports of a trace tree: their output, the SQL they run and their downloads.

`golden/` holds the output of the renderer as it was before it loaded the tree up front
and downloaded the images concurrently; these tests pin that the output didn't change.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlmodel import Session

from database.agent_graphs.models import AgentGraph
from database.agents.models import Agent, AgentType
from database.logging.models import (
    PENDING_IMAGE_UUID,
    AgentGraphNodeTrace,
    AgentGraphNodeTraceStatus,
    AgentGraphTrace,
    AgentGraphTraceStatus,
    AgentTrace,
    RobotException,
    SubAgentTrace,
    ToolTrace,
)
from database.provider.models import Router
from database.tools.models import Tool
from tests.unit.shared.auth_helpers import Headers
from tests.unit.shared.mock_s3 import MockS3Client

GOLDEN = Path(__file__).parent / "golden"
START = datetime(2026, 1, 1, tzinfo=UTC)


def _at(seconds: float) -> datetime:
    return START + timedelta(seconds=seconds)


def _stamp(seconds: float) -> str:
    """A message timestamp, as the agent hooks store it."""
    return _at(seconds).isoformat(sep=" ", timespec="microseconds")


def _id(n: int) -> UUID:
    return UUID(int=n)


@dataclass
class TraceTree:
    """A recovery's graph run over agent traces, with sub-agents two levels deep."""

    graph_trace_id: UUID
    root_trace_id: UUID
    image_keys: list[str]


async def _upload(content: bytes) -> str:
    return await MockS3Client.upload_bytes(content, "image/jpeg")


def _agent(session: Session, n: int, name: str, router: Router) -> Agent:
    agent = Agent(
        id=_id(n),
        name=name,
        description=f"{name} agent",
        prompt="You are a test agent.",
        input_type=Agent.InputType.TEXT,
        enabled=True,
        router=router,
        type=AgentType.Agent,
        arguments=[],
    )
    session.add(agent)
    return agent


def _trace(
    n: int, agent: Agent, seconds: float, messages: list[dict[str, Any]] | None
) -> AgentTrace:
    return AgentTrace(
        id=_id(n),
        agent_id=agent.id,
        inputs={"task": f"trace {n}"},
        messages=messages,
        cost=0.25 * n,
        created_at=_at(seconds),
        finished_at=_at(seconds + 50),
    )


def _tool_trace(n: int, trace: AgentTrace, tool: Tool, seconds: float) -> ToolTrace:
    return ToolTrace(
        id=_id(n),
        agent_trace_id=trace.id,
        tool_id=tool.id,
        input={"step": n},
        output=f"output {n}",
        success=n % 2 == 0,
        created_at=_at(seconds),
    )


@pytest_asyncio.fixture
async def trace_tree(session: Session, monkeypatch: pytest.MonkeyPatch) -> TraceTree:
    k1, k2, k3, k4, k5 = [
        await _upload(content)
        for content in (b"screen-1", b"screen-2", b"", b"screen-4", b"screen-5")
    ]
    monkeypatch.setattr(Tool, "get_tool_function", lambda _self: object())  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    router = Router(
        api_key="k",
        model_name="m",
        api_endpoint="https://api.test/v1",
        provider_type=Router.Provider.OPENAI,
    )
    gateway = _agent(session, 1, "Gateway", router)
    error_agent = _agent(session, 2, "Error Agent", router)
    gui_agent = _agent(session, 3, "GUI Agent", router)
    tool = Tool(id=_id(4), name="click", description="Click", fn_module="tests.x:y")
    exception = RobotException(id=_id(5), exception_details={"msg": "boom"})
    session.add_all([tool, exception])

    root = _trace(
        10,
        gateway,
        0,
        [
            {
                "role": "user",
                "content": [{"text": "Recover the robot"}, {"image": {"uuid": k1}}],
                "timestamp": _stamp(0),
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "toolResult": {
                            "content": [{"text": "clicked"}, {"image": {"uuid": k2}}]
                        }
                    },
                    {"image": {"uuid": PENDING_IMAGE_UUID}},
                ],
                "timestamp": _stamp(2),
            },
            {
                "role": "assistant",
                "content": [{"unknown": 1}, {"image": {"uuid": k3}}],
                "timestamp": _stamp(9),
            },
        ],
    )
    root.robot_exception_id = exception.id
    child = _trace(
        11,
        error_agent,
        3,
        [
            {
                "role": "assistant",
                "content": [{"text": "Diagnosed"}, {"image": {"uuid": k4}}],
                "timestamp": _stamp(4),
            }
        ],
    )
    grandchild = _trace(
        12,
        gui_agent,
        5.5,
        [{"role": "assistant", "content": [{"text": "Done"}], "timestamp": _stamp(6)}],
    )
    # A sub-agent without messages: its own sub-agents are not rendered.
    silent = _trace(13, gui_agent, 7, None)
    under_silent = _trace(
        14,
        gui_agent,
        8,
        [{"role": "user", "content": [{"image": {"uuid": "never-fetched"}}]}],
    )
    other_node = _trace(
        15,
        gui_agent,
        60,
        [
            {
                "role": "assistant",
                "content": [{"image": {"uuid": k5}}],
                "timestamp": _stamp(61),
            }
        ],
    )
    session.add_all([root, child, grandchild, silent, under_silent, other_node])
    session.flush()
    session.add_all(
        [
            _tool_trace(20, root, tool, 1),
            _tool_trace(21, child, tool, 5),
            _tool_trace(22, other_node, tool, 62),
            SubAgentTrace(id=_id(30), parent_trace_id=root.id, child_trace_id=child.id),
            SubAgentTrace(
                id=_id(31), parent_trace_id=child.id, child_trace_id=grandchild.id
            ),
            SubAgentTrace(
                id=_id(32), parent_trace_id=root.id, child_trace_id=silent.id
            ),
            SubAgentTrace(
                id=_id(33), parent_trace_id=silent.id, child_trace_id=under_silent.id
            ),
        ]
    )
    graph = AgentGraph(id=_id(40), name="UI Recovery", description="A test graph")
    graph_trace = AgentGraphTrace(
        id=_id(41),
        agent_graph_id=graph.id,
        robot_exception_id=exception.id,
        status=AgentGraphTraceStatus.FAILED,
        failure_reason="node limit",
        terminal_node_key="gui",
        answer={"ok": False},
        created_at=_at(0),
        finished_at=_at(100),
    )
    session.add_all([graph, graph_trace])
    session.flush()
    session.add_all(
        [
            AgentGraphNodeTrace(
                id=_id(50),
                agent_graph_trace_id=graph_trace.id,
                execution_index=0,
                node_key="gateway",
                agent_trace_id=root.id,
                status=AgentGraphNodeTraceStatus.COMPLETED,
            ),
            AgentGraphNodeTrace(
                id=_id(51),
                agent_graph_trace_id=graph_trace.id,
                execution_index=1,
                node_key="lost",
                status=AgentGraphNodeTraceStatus.FAILED,
                error="boom",
            ),
            AgentGraphNodeTrace(
                id=_id(52),
                agent_graph_trace_id=graph_trace.id,
                execution_index=2,
                node_key="gui",
                agent_trace_id=other_node.id,
                entry_condition={"recovered": True},
                status=AgentGraphNodeTraceStatus.COMPLETED,
            ),
        ]
    )
    session.commit()
    session.expunge_all()
    return TraceTree(
        graph_trace_id=_id(41), root_trace_id=_id(10), image_keys=[k1, k2, k3, k4, k5]
    )


@pytest.fixture
def export_session(engine: Engine) -> Iterator[Session]:
    """A session of its own, as a request's: nothing of the tree is loaded in it yet."""
    with Session(engine) as session:
        yield session


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The key lists of every bulk download; single downloads fail the test."""
    calls: list[list[str]] = []
    bulk_download = MockS3Client.bulk_download_bytes

    async def record(keys: list[str], bucket: str = "mock-bucket") -> dict[str, bytes]:
        calls.append(keys)
        return await bulk_download(keys, bucket)

    async def forbidden(key: str, bucket: str = "mock-bucket") -> bytes:
        raise AssertionError(f"downloaded {key} from {bucket} alone")

    monkeypatch.setattr(MockS3Client, "bulk_download_bytes", staticmethod(record))
    monkeypatch.setattr(MockS3Client, "download_bytes", staticmethod(forbidden))
    return calls


@pytest.mark.asyncio
async def test_rendering_a_loaded_tree_runs_no_sql_and_downloads_its_images_at_once(
    trace_tree: TraceTree,
    export_session: Session,
    statements: list[str],
    downloads: list[list[str]],
):
    trace = AgentGraphTrace.get_with_markdown_tree(
        export_session, trace_tree.graph_trace_id
    )
    assert trace is not None
    # The graph trace, its agent graph and node traces. Then per level of agent traces:
    # the traces, their agents, tool traces, tools and sub-traces; the third level has
    # no tool traces, so no tools. It grows with the depth, not the number of traces.
    assert len(statements) == 3 + 5 + 5 + 4
    assert not any("JOIN" in statement for statement in statements)
    statements.clear()

    _ = await trace.get_markdown_log()

    assert statements == []
    assert downloads == [trace_tree.image_keys]


@pytest.mark.parametrize("include_subtraces", [True, False])
@pytest.mark.asyncio
async def test_an_agent_trace_downloads_only_the_images_it_renders(
    trace_tree: TraceTree,
    export_session: Session,
    downloads: list[list[str]],
    include_subtraces: bool,
):
    trace = AgentTrace.get_with_markdown_tree(export_session, trace_tree.root_trace_id)
    assert trace is not None

    _ = await trace.get_markdown_log(include_subtraces=include_subtraces)

    k1, k2, k3, k4, _k5 = trace_tree.image_keys
    assert downloads == [[k1, k2, k3, k4] if include_subtraces else [k1, k2, k3]]


def test_an_unknown_trace_loads_as_none(export_session: Session):
    assert AgentTrace.get_with_markdown_tree(export_session, _id(999)) is None
    assert AgentGraphTrace.get_with_markdown_tree(export_session, _id(999)) is None


@pytest.mark.asyncio
async def test_the_markdown_endpoints_serve_the_exports(
    trace_tree: TraceTree, client: TestClient, user_headers: Headers
):
    agent_trace = client.get(
        f"/logging/markdown/?agent_trace_id={trace_tree.root_trace_id}",
        headers=user_headers,
    )
    graph_trace = client.get(
        f"/logging/agent_graph_traces/{trace_tree.graph_trace_id}/markdown",
        headers=user_headers,
    )

    assert agent_trace.status_code == graph_trace.status_code == status.HTTP_200_OK
    assert agent_trace.text == (GOLDEN / "agent_trace.md").read_text()
    assert graph_trace.text == (GOLDEN / "agent_graph_trace.md").read_text()


def test_the_agent_graph_trace_markdown_of_an_unknown_run_is_a_404(
    client: TestClient, user_headers: Headers
):
    response = client.get(
        f"/logging/agent_graph_traces/{_id(999)}/markdown", headers=user_headers
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
