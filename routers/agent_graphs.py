from collections.abc import Callable, Sequence
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from database.agent_graphs import service
from database.agent_graphs.models import (
    AgentGraph,
    AgentGraphCreate,
    AgentGraphEdge,
    AgentGraphEdgeCreate,
    AgentGraphEdgeRead,
    AgentGraphEdgeUpdate,
    AgentGraphNode,
    AgentGraphNodeCreate,
    AgentGraphNodeRead,
    AgentGraphNodeUpdate,
    AgentGraphRead,
    AgentGraphUpdate,
)
from database.auth.models import User
from database.engine import commit_or_500
from database.general import SessionDep
from middlewares.auth import get_current_user, require_admin

router = APIRouter(
    prefix="/agent_graphs",
    tags=["Agent Graphs"],
    dependencies=[Depends(get_current_user)],
)

AdminDep = Annotated[User, Depends(require_admin)]

_REJECTIONS: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "A referenced row doesn't exist."},
    status.HTTP_400_BAD_REQUEST: {
        "description": "The write breaks a rule; the detail lists the violations."
    },
}


def _write[T](session: Session, action: str, write: Callable[[], T]) -> T:
    """Run a service write and commit it, or roll it back and answer its rejection.

    The service checks the unique constraints itself, so one that still fails at
    flush lost a race with a concurrent write, and is rejected like the service
    would have.
    """
    try:
        result = write()
        session.flush()
    except service.AgentGraphError as exc:
        session.rollback()
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc
    except IntegrityError as exc:
        session.rollback()
        message = (
            "The write conflicts with a concurrent change to this agent graph, such "
            "as a duplicate name, node key or edge; reload the graph and retry."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": message, "violations": [message]},
        ) from exc
    commit_or_500(session, action)
    return result


def _graph(session: Session, graph_id: UUID) -> AgentGraph:
    graph = session.get(AgentGraph, graph_id)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent graph not found"
        )
    return graph


def _node(graph: AgentGraph, node_id: UUID) -> AgentGraphNode:
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent graph node not found"
        )
    return node


def _edge(graph: AgentGraph, edge_id: UUID) -> AgentGraphEdge:
    edge = next((e for e in graph.edges if e.id == edge_id), None)
    if edge is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Agent graph edge not found"
        )
    return edge


def _read(session: Session, graph: AgentGraph) -> AgentGraphRead:
    session.refresh(graph)
    return _as_read(graph)


def _as_read(graph: AgentGraph) -> AgentGraphRead:
    return AgentGraphRead.model_validate(graph, from_attributes=True).model_copy(
        update={"warnings": service.warnings(graph)}
    )


@router.get(
    "/", response_model=Sequence[AgentGraphRead], summary="List all agent graphs"
)
def list_agent_graphs(session: SessionDep) -> Sequence[AgentGraphRead]:
    return [_as_read(graph) for graph in session.exec(select(AgentGraph)).all()]


@router.get(
    "/{graph_id}", response_model=AgentGraphRead, summary="Get agent graph by ID"
)
def get_agent_graph(graph_id: UUID, session: SessionDep) -> AgentGraphRead:
    return _as_read(_graph(session, graph_id))


@router.post(
    "/",
    response_model=AgentGraphRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create agent graph",
    responses=_REJECTIONS,
)
def create_agent_graph(
    payload: AgentGraphCreate, session: SessionDep, _admin: AdminDep
) -> AgentGraphRead:
    graph = _write(
        session,
        "create agent graph",
        lambda: service.create_graph(session, payload),
    )
    return _read(session, graph)


@router.patch(
    "/{graph_id}",
    response_model=AgentGraphRead,
    summary="Update agent graph",
    description="Enabling a graph, or changing an enabled one, validates it.",
    responses=_REJECTIONS,
)
def update_agent_graph(
    graph_id: UUID, payload: AgentGraphUpdate, session: SessionDep, _admin: AdminDep
) -> AgentGraphRead:
    graph = _graph(session, graph_id)
    _ = _write(
        session,
        "update agent graph",
        lambda: service.update_graph(session, graph, payload),
    )
    return _read(session, graph)


@router.delete(
    "/{graph_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete agent graph with its nodes and edges",
)
def delete_agent_graph(graph_id: UUID, session: SessionDep, _admin: AdminDep) -> None:
    graph = _graph(session, graph_id)
    _write(session, "delete agent graph", lambda: service.delete_graph(session, graph))


@router.post(
    "/{graph_id}/nodes",
    response_model=AgentGraphNodeRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a node to an agent graph",
    responses=_REJECTIONS,
)
def add_agent_graph_node(
    graph_id: UUID,
    payload: AgentGraphNodeCreate,
    session: SessionDep,
    _admin: AdminDep,
) -> AgentGraphNode:
    graph = _graph(session, graph_id)
    node = _write(
        session,
        "add agent graph node",
        lambda: service.add_node(session, graph, payload),
    )
    session.refresh(node)
    return node


@router.patch(
    "/{graph_id}/nodes/{node_id}",
    response_model=AgentGraphNodeRead,
    summary="Change a node of an agent graph",
    responses=_REJECTIONS,
)
def update_agent_graph_node(
    graph_id: UUID,
    node_id: UUID,
    payload: AgentGraphNodeUpdate,
    session: SessionDep,
    _admin: AdminDep,
) -> AgentGraphNode:
    graph = _graph(session, graph_id)
    node = _node(graph, node_id)
    _ = _write(
        session,
        "update agent graph node",
        lambda: service.update_node(session, graph, node, payload),
    )
    session.refresh(node)
    return node


@router.delete(
    "/{graph_id}/nodes/{node_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a node and its edges from an agent graph",
    responses=_REJECTIONS,
)
def remove_agent_graph_node(
    graph_id: UUID, node_id: UUID, session: SessionDep, _admin: AdminDep
) -> None:
    graph = _graph(session, graph_id)
    node = _node(graph, node_id)
    _write(
        session,
        "remove agent graph node",
        lambda: service.remove_node(session, graph, node),
    )


@router.post(
    "/{graph_id}/edges",
    response_model=AgentGraphEdgeRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add an edge to an agent graph",
    responses=_REJECTIONS,
)
def add_agent_graph_edge(
    graph_id: UUID,
    payload: AgentGraphEdgeCreate,
    session: SessionDep,
    _admin: AdminDep,
) -> AgentGraphEdge:
    graph = _graph(session, graph_id)
    edge = _write(
        session,
        "add agent graph edge",
        lambda: service.add_edge(session, graph, payload),
    )
    session.refresh(edge)
    return edge


@router.patch(
    "/{graph_id}/edges/{edge_id}",
    response_model=AgentGraphEdgeRead,
    summary="Change an edge of an agent graph",
    responses=_REJECTIONS,
)
def update_agent_graph_edge(
    graph_id: UUID,
    edge_id: UUID,
    payload: AgentGraphEdgeUpdate,
    session: SessionDep,
    _admin: AdminDep,
) -> AgentGraphEdge:
    graph = _graph(session, graph_id)
    edge = _edge(graph, edge_id)
    _ = _write(
        session,
        "update agent graph edge",
        lambda: service.update_edge(session, graph, edge, payload),
    )
    session.refresh(edge)
    return edge


@router.delete(
    "/{graph_id}/edges/{edge_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an edge from an agent graph",
    responses=_REJECTIONS,
)
def remove_agent_graph_edge(
    graph_id: UUID, edge_id: UUID, session: SessionDep, _admin: AdminDep
) -> None:
    graph = _graph(session, graph_id)
    edge = _edge(graph, edge_id)
    _write(
        session,
        "remove agent graph edge",
        lambda: service.remove_edge(session, graph, edge),
    )
