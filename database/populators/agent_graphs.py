"""Populator for the agent graph recoveries run: "UI Recovery".

The gateway classifies the failure; only a UI exception goes on to the Recovery
Inferer, the Tooled GuiAgent and the Continuation Inferer. Agents must have been
populated first (populate_agents). Idempotent: the graph is matched by name.

The agent populator never overwrites an agent, so a database seeded before this
graph keeps its gateway's old prompt and response model. There the graph is created
disabled, recoveries keep running the gateway alone ("Default Recovery"), and the
administrator API calls that migrate the database are printed on every run until
they are made, as are the ones left when it is migrated halfway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, or_, select

from database.agent_graphs import service
from database.agent_graphs.conditions import Compare, Condition
from database.agent_graphs.models import (
    AgentGraph,
    AgentGraphCreate,
    AgentGraphEdgeCreate,
    AgentGraphNode,
    AgentGraphNodeCreate,
    AgentGraphUpdate,
)
from database.agents.models import Agent
from database.populators.agents import DEFAULT_PROMPTS_DIR

LOG_PREFIX = "[populate_agent_graphs]"
_REPO_ROOT = Path(__file__).resolve().parents[2]

UI_RECOVERY_GRAPH = "UI Recovery"
UI_RECOVERY_DESCRIPTION = (
    "The gateway classifies the failure. A UI exception goes on to the Recovery "
    "Inferer, which writes the recovery task, the Tooled GuiAgent, which carries it "
    "out on the robot's desktop, and the Continuation Inferer, which tells the robot "
    "where to continue. Any other failure ends at the gateway, without success."
)
UI_RECOVERY_MAX_NODE_EXECUTIONS = 4

GATEWAY = "Gateway Orchestrator"
GATEWAY_DECISION = "templates.gateway.GatewayDecision"
GATEWAY_PROMPT_FILE = "gateway.md"
UI_EXCEPTION_HANDLER = "UI Exception Handler"

ENTRY_NODE = "gateway"
NODES: tuple[tuple[str, str, dict[str, str]], ...] = (
    (ENTRY_NODE, GATEWAY, {}),
    ("recovery_inferer", "Recovery Inferer", {}),
    ("tooled_guiagent", "Tooled GuiAgent", {"task": "recovery_inferer.recovery_task"}),
    (
        "continuation_inferer",
        "Continuation Inferer",
        {
            "gui_summary": "tooled_guiagent.summary",
            "goal_reached": "tooled_guiagent.goal_reached",
        },
    ),
)
EDGES: tuple[tuple[str, str, Condition | None], ...] = (
    (
        ENTRY_NODE,
        "recovery_inferer",
        Compare(field="ui_exception", op="eq", value=True),
    ),
    ("recovery_inferer", "tooled_guiagent", None),
    ("tooled_guiagent", "continuation_inferer", None),
)

LEGACY_GRAPH = "Default Recovery"
LEGACY_DESCRIPTION = "Runs the gateway agent on every recovery."


@dataclass(frozen=True, slots=True)
class ManualStep:
    """An administrator API call that moves an old database to "UI Recovery"."""

    action: str
    method: str
    path: str
    body: dict[str, Any]
    shown_body: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        shown = self.body | self.shown_body
        return f"{self.action}:\n     {self.method} {self.path} {json.dumps(shown)}"


def populate_agent_graphs(engine: Engine) -> list[ManualStep]:
    """Create "UI Recovery", enabled as the default recovery graph if it can be.

    Returns the steps left to migrate the database, which are also printed; none
    once it is migrated, or while another graph is the enabled default.
    """
    with Session(engine) as session:
        gateway = _agent(session, GATEWAY)
        if gateway is None:
            print(f"{LOG_PREFIX} No agent '{GATEWAY}'; no agent graph created.")
            return []
        migrated = gateway.response_model == GATEWAY_DECISION
        if not migrated:
            _ensure_legacy_graph(session, gateway)
        graph = _graph(session, UI_RECOVERY_GRAPH) or _create_ui_recovery(
            session, enable=migrated
        )
        if graph is None or graph.enabled:
            return []
        if migrated:
            if AgentGraph.default_recovery_id(session) is not None:
                return []
            why = (
                "no agent graph is the enabled default recovery graph, so recoveries "
                "end with NO_AGENT_GRAPH"
            )
        else:
            why = (
                f"the gateway '{gateway.name}' still has the response model "
                f"{gateway.response_model}, so recoveries run the gateway alone"
            )
        entered = _graphs_entered(session, gateway, graph)
        if any(other.is_default_recovery for other in entered):
            why += (
                "; from the first call to the last, recoveries end with NO_AGENT_GRAPH"
            )
        steps = _manual_steps(session, gateway, graph, entered, migrated=migrated)
        print(
            f"{LOG_PREFIX} The agent graph '{graph.name}' is disabled: {why}. To "
            f"recover through '{graph.name}', make these API calls as an "
            "administrator, in order:"
        )
        for number, step in enumerate(steps, start=1):
            print(f"  {number}. {step}")
        return steps


def _create_ui_recovery(session: Session, *, enable: bool) -> AgentGraph | None:
    """Create the graph as a draft, then enable it if ``enable`` and it passes."""
    found = {name: _agent(session, name) for _, name, _ in NODES}
    agents = {name: agent for name, agent in found.items() if agent is not None}
    missing = [name for name in found if name not in agents]
    if missing:
        print(
            f"{LOG_PREFIX} No agent {', '.join(repr(name) for name in missing)}; no "
            f"'{UI_RECOVERY_GRAPH}' graph created."
        )
        return None
    graph = service.create_graph(
        session,
        AgentGraphCreate(
            name=UI_RECOVERY_GRAPH,
            description=UI_RECOVERY_DESCRIPTION,
            max_node_executions=UI_RECOVERY_MAX_NODE_EXECUTIONS,
        ),
    )
    nodes: dict[str, AgentGraphNode] = {}
    for key, name, input_map in NODES:
        nodes[key] = service.add_node(
            session,
            graph,
            AgentGraphNodeCreate(
                key=key,
                agent_id=agents[name].id,
                is_entry=key == ENTRY_NODE,
                input_map=input_map,
            ),
        )
    for source, target, condition in EDGES:
        _ = service.add_edge(
            session,
            graph,
            AgentGraphEdgeCreate(
                source_node_id=nodes[source].id,
                target_node_id=nodes[target].id,
                condition=condition,
            ),
        )
    session.commit()
    print(f"{LOG_PREFIX} Created the agent graph '{UI_RECOVERY_GRAPH}'.")
    if enable:
        _enable(session, graph)
    return graph


def _enable(session: Session, graph: AgentGraph) -> None:
    """Enable ``graph`` as the default recovery graph, unless another one is."""
    default = session.exec(
        select(AgentGraph.name).where(col(AgentGraph.is_default_recovery).is_(True))
    ).first()
    if default is not None:
        print(
            f"{LOG_PREFIX} '{graph.name}' stays disabled: '{default}' is the default "
            "recovery graph."
        )
        return
    try:
        _ = service.update_graph(
            session, graph, AgentGraphUpdate(enabled=True, is_default_recovery=True)
        )
    except service.AgentGraphError as exc:
        session.rollback()
        print(f"{LOG_PREFIX} '{graph.name}' stays disabled: {exc.message}")
        for violation in exc.violations:
            print(f"  - {violation}")
        return
    session.commit()
    print(f"{LOG_PREFIX} Enabled '{graph.name}' as the default recovery graph.")


def _ensure_legacy_graph(session: Session, gateway: Agent) -> None:
    """Keep recoveries running the gateway alone, as before "UI Recovery".

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
            key=ENTRY_NODE,
            agent_id=gateway.id,
            is_entry=True,
        )
    ]
    session.add(graph)
    session.commit()
    print(f"{LOG_PREFIX} Created the agent graph '{LEGACY_GRAPH}': the gateway alone.")


def _graphs_entered(
    session: Session, gateway: Agent, graph: AgentGraph
) -> list[AgentGraph]:
    """The enabled graphs other than ``graph`` that start at ``gateway``."""
    return list(
        session.exec(
            select(AgentGraph)
            .join(AgentGraphNode)
            .where(
                AgentGraphNode.agent_id == gateway.id,
                col(AgentGraphNode.is_entry).is_(True),
                col(AgentGraph.enabled).is_(True),
                AgentGraph.id != graph.id,
            )
            .order_by(col(AgentGraph.name))
        ).all()
    )


def _manual_steps(
    session: Session,
    gateway: Agent,
    graph: AgentGraph,
    entered: list[AgentGraph],
    *,
    migrated: bool,
) -> list[ManualStep]:
    """The API calls left, in order, that make ``graph`` the default recovery graph."""
    steps = [
        ManualStep(
            f"Disable the agent graph '{other.name}', which the gateway enters",
            "PATCH",
            f"/agent_graphs/{other.id}",
            {"enabled": False},
        )
        for other in entered
    ]
    if not migrated:
        prompt_path = DEFAULT_PROMPTS_DIR / GATEWAY_PROMPT_FILE
        steps.append(
            ManualStep(
                f"Give the gateway '{gateway.name}' its classification prompt and "
                "response model",
                "PATCH",
                f"/agents/{gateway.id}",
                {
                    "prompt": prompt_path.read_text(encoding="utf-8"),
                    "response_model": GATEWAY_DECISION,
                },
                {"prompt": f"<the text of {prompt_path.relative_to(_REPO_ROOT)}>"},
            )
        )
    handler = _agent(session, UI_EXCEPTION_HANDLER)
    if handler is not None and handler.enabled:
        steps.append(
            ManualStep(
                f"Disable the agent '{handler.name}', which the graph replaces",
                "PATCH",
                f"/agents/{handler.id}",
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
