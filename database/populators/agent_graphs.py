"""JSON-driven populator for agent graphs: ``agent_graphs.json``.

Agents must have been populated first (populate_agents); nodes and replaced agents
name them. Idempotent: graphs are matched by name. A graph seeded enabled is enabled
once it passes the same validation as the API, and one seeded as the default
recovery graph only takes over when no graph is the default yet.

The agent populator never overwrites an agent, so on a database seeded earlier a
graph's entry agent can still have an older response model than its seed: the
gateway, before "UI Recovery". There the graph is created disabled, recoveries keep
running the entry agent alone ("Default Recovery"), and the administrator API calls
that migrate the database are printed on every run until they are made, as are the
ones left when it is migrated halfway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Engine
from sqlmodel import Field, Session, col, or_, select

from database.agent_graphs import service
from database.agent_graphs.conditions import Condition
from database.agent_graphs.models import (
    AgentGraph,
    AgentGraphCreate,
    AgentGraphEdgeCreate,
    AgentGraphNode,
    AgentGraphNodeCreate,
    AgentGraphUpdate,
)
from database.agents.models import Agent
from database.agents.specs import OutputModelSpec
from database.populators.agents import DEFAULT_PROMPTS_DIR, agent_seed
from database.populators.config import load_config

LOG_PREFIX = "[populate_agent_graphs]"
CONFIG_PATH = Path(__file__).parent / "agent_graphs.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]

LEGACY_GRAPH = "Default Recovery"
LEGACY_DESCRIPTION = "Runs the gateway agent on every recovery."
LEGACY_NODE = "gateway"


class NodeSeed(BaseModel):
    key: str
    agent: str
    is_entry: bool = False
    input_map: dict[str, str] = Field(default_factory=dict)


class EdgeSeed(BaseModel):
    source: str
    target: str
    priority: int = 0
    condition: Condition | None = None


class AgentGraphSeed(AgentGraphCreate):
    """An agent graph of ``agent_graphs.json``, with its agents named."""

    replaces: list[str] = Field(
        default_factory=list,
        description="Agents the graph takes over from, disabled when migrating to it.",
    )
    nodes: list[NodeSeed]
    edges: list[EdgeSeed] = Field(default_factory=list)

    @property
    def entry(self) -> NodeSeed:
        return next(node for node in self.nodes if node.is_entry)


@dataclass(frozen=True, slots=True)
class ManualStep:
    """An administrator API call that migrates an old database to a seeded graph."""

    action: str
    method: str
    path: str
    body: dict[str, Any]
    shown_body: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        shown = self.body | self.shown_body
        return f"{self.action}:\n     {self.method} {self.path} {json.dumps(shown)}"


def populate_agent_graphs(engine: Engine) -> list[ManualStep]:
    """Create the seeded graphs that are missing.

    Returns the steps left to migrate the database, which are also printed; none
    once it is migrated, or while another graph is the enabled default.
    """
    with Session(engine) as session:
        return [
            step for seed in _load_seeds() for step in _populate_graph(session, seed)
        ]


def _load_seeds() -> list[AgentGraphSeed]:
    config = load_config(CONFIG_PATH, "populate_agent_graphs")
    return [AgentGraphSeed.model_validate(g) for g in config.get("agent_graphs", [])]


def _populate_graph(session: Session, seed: AgentGraphSeed) -> list[ManualStep]:
    entry = _agent(session, seed.entry.agent)
    if entry is None:
        print(
            f"{LOG_PREFIX} No agent '{seed.entry.agent}'; no '{seed.name}' graph "
            "created."
        )
        return []
    outdated = _outdated(entry)
    if outdated and seed.is_default_recovery:
        _ensure_legacy_graph(session, entry)
    graph = _graph(session, seed.name) or _create_graph(
        session, seed, enable=seed.enabled and not outdated
    )
    if graph is None or graph.enabled:
        return []
    if not (seed.enabled and seed.is_default_recovery):
        return []
    why = _disabled_reason(session, entry, outdated=outdated)
    if why is None:
        return []
    entered = _graphs_entered(session, entry, graph)
    if any(other.is_default_recovery for other in entered):
        why += "; from the first call to the last, recoveries end with NO_AGENT_GRAPH"
    steps = _manual_steps(session, seed, entry, graph, entered, outdated=outdated)
    print(
        f"{LOG_PREFIX} The agent graph '{graph.name}' is disabled: {why}. To "
        f"recover through '{graph.name}', make these API calls as an "
        "administrator, in order:"
    )
    for number, step in enumerate(steps, start=1):
        print(f"  {number}. {step}")
    return steps


def _disabled_reason(session: Session, entry: Agent, *, outdated: bool) -> str | None:
    """Why recoveries don't run the seeded default graph, or None if that is fine:
    another graph is the enabled default."""
    if outdated:
        current = entry.output_model.name if entry.output_model else "none"
        return (
            f"its entry agent '{entry.name}' has another output model ({current}) "
            "than agents.json seeds, so recoveries run that agent alone"
        )
    if AgentGraph.default_recovery_id(session) is None:
        return (
            "no agent graph is the enabled default recovery graph, so recoveries "
            "end with NO_AGENT_GRAPH"
        )
    return None


def _outdated(agent: Agent) -> bool:
    """Whether ``agent`` has another output model than ``agents.json`` seeds."""
    seed = agent_seed(agent.name)
    if seed is None:
        return False
    seeded = seed.get("output_model")
    expected = None if seeded is None else OutputModelSpec.model_validate(seeded)
    return agent.output_model != expected


def _create_graph(
    session: Session, seed: AgentGraphSeed, *, enable: bool
) -> AgentGraph | None:
    """Create the graph as a draft, then enable it if ``enable`` and it passes."""
    found = {node.agent: _agent(session, node.agent) for node in seed.nodes}
    agents = {name: agent for name, agent in found.items() if agent is not None}
    missing = [name for name in found if name not in agents]
    if missing:
        print(
            f"{LOG_PREFIX} No agent {', '.join(repr(name) for name in missing)}; no "
            f"'{seed.name}' graph created."
        )
        return None
    settings = seed.model_dump(include=set(AgentGraphCreate.model_fields))
    graph = service.create_graph(
        session,
        AgentGraphCreate.model_validate(
            settings | {"enabled": False, "is_default_recovery": False}
        ),
    )
    nodes: dict[str, AgentGraphNode] = {}
    for node in seed.nodes:
        nodes[node.key] = service.add_node(
            session,
            graph,
            AgentGraphNodeCreate(
                key=node.key,
                agent_id=agents[node.agent].id,
                is_entry=node.is_entry,
                input_map=node.input_map,
            ),
        )
    for edge in seed.edges:
        _ = service.add_edge(
            session,
            graph,
            AgentGraphEdgeCreate(
                source_node_id=nodes[edge.source].id,
                target_node_id=nodes[edge.target].id,
                priority=edge.priority,
                condition=edge.condition,
            ),
        )
    session.commit()
    print(f"{LOG_PREFIX} Created the agent graph '{seed.name}'.")
    if enable:
        _enable(session, graph, make_default=seed.is_default_recovery)
    return graph


def _enable(session: Session, graph: AgentGraph, *, make_default: bool) -> None:
    """Enable ``graph``, as the default recovery graph unless another one is."""
    if make_default:
        default = session.exec(
            select(AgentGraph.name).where(col(AgentGraph.is_default_recovery).is_(True))
        ).first()
        if default is not None:
            print(
                f"{LOG_PREFIX} '{graph.name}' stays disabled: '{default}' is the "
                "default recovery graph."
            )
            return
    update = (
        AgentGraphUpdate(enabled=True, is_default_recovery=True)
        if make_default
        else AgentGraphUpdate(enabled=True)
    )
    try:
        _ = service.update_graph(session, graph, update)
    except service.AgentGraphError as exc:
        session.rollback()
        print(f"{LOG_PREFIX} '{graph.name}' stays disabled: {exc.message}")
        for violation in exc.violations:
            print(f"  - {violation}")
        return
    session.commit()
    default = " as the default recovery graph" if make_default else ""
    print(f"{LOG_PREFIX} Enabled '{graph.name}'{default}.")


def _ensure_legacy_graph(session: Session, entry: Agent) -> None:
    """Keep recoveries running ``entry`` alone, as before the seeded graph.

    Nothing is created while a default recovery graph, or a graph with the legacy
    graph's name, exists. The graph is written directly: the service rejects a
    default recovery graph that has no GUI node.
    """
    existing = session.exec(
        select(AgentGraph.id).where(
            or_(
                col(AgentGraph.is_default_recovery).is_(True),
                AgentGraph.name == LEGACY_GRAPH,
            )
        )
    ).first()
    if existing is not None:
        return
    graph = AgentGraph(
        name=LEGACY_GRAPH,
        description=LEGACY_DESCRIPTION,
        enabled=True,
        is_default_recovery=True,
    )
    graph.nodes = [
        AgentGraphNode(
            agent_graph_id=graph.id,
            key=LEGACY_NODE,
            agent_id=entry.id,
            is_entry=True,
        )
    ]
    session.add(graph)
    session.commit()
    print(
        f"{LOG_PREFIX} Created the agent graph '{LEGACY_GRAPH}': '{entry.name}' alone."
    )


def _graphs_entered(
    session: Session, entry: Agent, graph: AgentGraph
) -> list[AgentGraph]:
    """The enabled graphs other than ``graph`` that start at ``entry``."""
    return list(
        session.exec(
            select(AgentGraph)
            .join(AgentGraphNode)
            .where(
                AgentGraphNode.agent_id == entry.id,
                col(AgentGraphNode.is_entry).is_(True),
                col(AgentGraph.enabled).is_(True),
                AgentGraph.id != graph.id,
            )
            .order_by(col(AgentGraph.name))
        ).all()
    )


def _manual_steps(
    session: Session,
    seed: AgentGraphSeed,
    entry: Agent,
    graph: AgentGraph,
    entered: list[AgentGraph],
    *,
    outdated: bool,
) -> list[ManualStep]:
    """The API calls left, in order, that make ``graph`` the default recovery graph."""
    steps = [
        ManualStep(
            f"Disable the agent graph '{other.name}', which '{entry.name}' enters",
            "PATCH",
            f"/agent_graphs/{other.id}",
            {"enabled": False},
        )
        for other in entered
    ]
    entry_seed = agent_seed(entry.name)
    if outdated and entry_seed is not None:
        prompt_path = DEFAULT_PROMPTS_DIR / entry_seed["prompt_file"]
        steps.append(
            ManualStep(
                f"Give the agent '{entry.name}' its seeded prompt and output model",
                "PATCH",
                f"/agents/{entry.id}",
                {
                    "prompt": prompt_path.read_text(encoding="utf-8"),
                    "output_model": entry_seed["output_model"],
                },
                {"prompt": f"<the text of {prompt_path.relative_to(_REPO_ROOT)}>"},
            )
        )
    for name in seed.replaces:
        replaced = _agent(session, name)
        if replaced is not None and replaced.enabled:
            steps.append(
                ManualStep(
                    f"Disable the agent '{replaced.name}', which the graph replaces",
                    "PATCH",
                    f"/agents/{replaced.id}",
                    {"enabled": False},
                )
            )
    steps.append(
        ManualStep(
            f"Enable the agent graph '{graph.name}' as the default recovery graph",
            "PATCH",
            f"/agent_graphs/{graph.id}",
            {"enabled": True, "is_default_recovery": True},
        )
    )
    return steps


def _agent(session: Session, name: str) -> Agent | None:
    return session.exec(select(Agent).where(Agent.name == name)).first()


def _graph(session: Session, name: str) -> AgentGraph | None:
    return session.exec(select(AgentGraph).where(AgentGraph.name == name)).first()
