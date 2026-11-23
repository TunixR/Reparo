"""
JSON-driven populator for registering framework Agents into the database.

This replaces the previous hardcoded definitions and loads agent configurations
from a JSON file. Tool populators and router populators remain unchanged.

Notes:
- Idempotent: existing agents are matched by name and never overwritten.
- Tools must have been populated previously (populate_tools) or they will be skipped.
- Each agent names a router role from routers.json (`router`); populate_routers must run first.
- Prompts are read from Markdown files in `default_prompts/`, named by each agent's `prompt_file`.
- Arguments and output models are inline data (`FieldSpec`, `OutputModelSpec` in
  `database/agents/specs.py`), validated when the agent is built.
- Supports sub-agent relationships and tool attachments via JSON.
- Each agent passes its type's rules (`validate_agent`) before it is saved; the
  entry's fields its type ignores are printed. A GUI agent takes the fixed GUI inputs, so its entry
  leaves out ``arguments`` and ``input_type``.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, select

from database.agents.behaviours import validate_agent
from database.agents.models import (
    Agent,
    AgentTool,
    AgentType,
    SubAgent,
)
from database.populators.routers import find_router
from database.provider.models import Router
from database.tools.models import Tool


def _attach_tools(
    session: Session,
    agent: Agent,
    tool_names: list[str],
    limits: list[int | None] | None = None,
    required: list[bool] | None = None,
) -> None:
    """
    Attach tools (by name) to an Agent via AgentTool association.
    Skips missing tools but logs a warning.
    When limits or required flags are provided, they must match the number of tools;
    each tool gets its corresponding value.
    """

    if limits is not None and len(tool_names) != len(limits):
        raise ValueError(
            f"[populate_agents] Tools/limits length mismatch for agent '{agent.name}': \
            {len(tool_names)} tools vs {len(limits)} limits."
        )
    if required is not None and len(tool_names) != len(required):
        raise ValueError(
            f"[populate_agents] Tools/required length mismatch for agent '{agent.name}': "
            f"{len(tool_names)} tools vs {len(required)} required flags."
        )

    for idx, tool_name in enumerate(tool_names):
        tool = session.exec(select(Tool).where(Tool.name == tool_name)).first()
        if not tool:
            print(
                f"[populate_agents] Warning: Tool '{tool_name}' not found. Skipping for agent '{agent.name}'."
            )
            continue

        # Check if already associated
        if any(at.tool.id == tool.id for at in agent.tools):
            continue

        session.add(
            AgentTool(
                agent=agent,
                agent_id=agent.id,
                tool=tool,
                tool_id=tool.id,
                limit=(limits[idx] if limits is not None else None),
                required=(required[idx] if required is not None else False),
            )
        )


def _create_agent(
    session: Session,
    name: str,
    description: str,
    prompt: str,
    arguments: list[dict[str, Any]],
    output_model: dict[str, Any] | None,
    router: Router,
    agent_type: AgentType,
    input_type: Agent.InputType = Agent.InputType.TEXT,
    enabled: bool = True,
    reasoning_effort: Agent.ReasoningEffort | None = None,
    parser_path: str | None = None,
    given: Collection[str] = (),
) -> Agent:
    """Create and persist an Agent if it doesn't already exist.

    Prints the fields in ``given`` (the entry's keys) that the agent's type ignores.
    """
    existing = session.exec(select(Agent).where(Agent.name == name)).first()
    if existing:
        return existing

    agent = Agent(
        name=name,
        description=description,
        prompt=prompt,
        arguments=arguments,
        output_model=output_model,
        input_type=input_type,
        enabled=enabled,
        reasoning_effort=reasoning_effort,
        router=router,
        type=agent_type,
        parser_path=parser_path,
    )
    ignored = validate_agent(agent)
    session.add(agent)
    session.commit()
    session.refresh(agent)
    print(f"[populate_agents] Created agent '{name}'.")
    for field in ignored:
        if field.field in given:
            print(
                f"[populate_agents]   '{name}' ignores {field.field}: {field.reason}."
            )
    return agent


def _create_sub_agent(
    session: Session, parent: Agent, child: Agent, limit: int | None = None
) -> None:
    """Create a SubAgent relationship if not present."""
    existing = session.exec(
        select(SubAgent).where(
            SubAgent.parent_agent == parent, SubAgent.child_agent == child
        )
    ).first()
    if existing:
        return
    session.add(
        SubAgent(
            parent_agent_id=parent.id,
            parent_agent=parent,
            child_agent_id=child.id,
            child_agent=child,
            limit=limit,
        )
    )


def _resolve_input_type(value: str | None) -> Agent.InputType:
    if not value:
        return Agent.InputType.TEXT
    mapping = {
        "TEXT": Agent.InputType.TEXT,
        "IMAGETEXT": Agent.InputType.IMAGETEXT,
    }
    return mapping.get(value.upper(), Agent.InputType.TEXT)


def _resolve_reasoning_effort(value: str | None) -> Agent.ReasoningEffort | None:
    """Resolve a configured reasoning effort, if any."""
    if not value:
        return None
    try:
        return Agent.ReasoningEffort(value.lower())
    except ValueError as e:
        raise ValueError(f"Unknown reasoning effort '{value}'.") from e


def _resolve_agent_type(value: str) -> AgentType:
    try:
        return AgentType(value)
    except ValueError as e:
        raise ValueError(f"Unknown AgentType '{value}'.") from e


def _load_json_config() -> dict[str, Any]:
    """
    Load agent configuration from adjacent JSON file:
    database/populators/agents.json

    Expected structure:
    {
      "agents": [
        {
          "name": "...",
          "description": "...",
          "prompt_file": "agent_name.md",  // in default_prompts/
          "router": "text" | "vision" | "vision_tool" | "uitars" | "grounding",  // a role in routers.json
          "agent_type": "Agent" | "TooledGuiAgent" | "TextGuiAgent",
          "parser_path": "module:attribute",  // a TextGuiAgent's parser, else absent
          "input_type": "TEXT" | "IMAGETEXT",  // a GUI agent's is always IMAGETEXT
          "enabled": true,
          "reasoning_effort": "medium",  // optional, unset keeps the provider default
          "arguments": [  // FieldSpec, in order; a "default" makes one optional; a GUI agent has fixed ones
            {"name": "...", "type": "str | None", "description": "...", "default": null}
          ],
          "output_model": {  // OutputModelSpec, or null; a GUI agent ignores it
            "name": "ClassName",  // the structured-output tool name the LLM sees
            "description": "...",  // the tool description
            "fields": [{"name": "...", "type": "Literal['a', 'b']", "description": "..."}]
          },
          "tools": {"names": ["take_screenshot"], "limits": [null], "required": [false]}
          }
        ],
      "sub_agents": [
        {"parent": "UI Exception Handler", "child": "UI Standalone UITARS", "limit": 1}
      ]
    }
    """
    here = Path(__file__).parent
    cfg_path = here / "agents.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"[populate_agents] Missing JSON configuration file: {cfg_path}"
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


DEFAULT_PROMPTS_DIR = Path(__file__).parent / "default_prompts"


def agent_seed(name: str) -> dict[str, Any] | None:
    """The configuration ``agents.json`` seeds the agent called ``name`` with."""
    agents: list[dict[str, Any]] = _load_json_config().get("agents", [])
    return next((agent for agent in agents if agent["name"] == name), None)


def _read_prompt(agent_name: str, prompt_file: str | None) -> str:
    """Read an agent's prompt from its file in ``default_prompts/``."""
    if not prompt_file:
        raise ValueError(f"[populate_agents] Agent '{agent_name}' has no prompt_file.")
    path = DEFAULT_PROMPTS_DIR / prompt_file
    if path.parent != DEFAULT_PROMPTS_DIR:
        raise ValueError(
            f"[populate_agents] prompt_file '{prompt_file}' of agent '{agent_name}' "
            f"must be a file name in {DEFAULT_PROMPTS_DIR}."
        )
    return path.read_text(encoding="utf-8")


def populate_agents(engine: Engine) -> None:
    """
    Populate the database with framework agents and their relationships from JSON.
    """
    cfg = _load_json_config()
    agents_cfg: list[dict[str, Any]] = cfg.get("agents", [])
    subs_cfg: list[dict[str, Any]] = cfg.get("sub_agents", [])

    if not agents_cfg:
        print("[populate_agents] No agents found in JSON configuration.")
        return

    with Session(engine) as session:
        # Create agents
        created_agents: dict[str, Agent] = {}

        for item in agents_cfg:
            name = item["name"]
            description = item.get("description", "") or ""
            prompt = _read_prompt(name, item.get("prompt_file"))

            router = find_router(session, item.get("router", "text"))

            agent_type = _resolve_agent_type(item["agent_type"])
            input_type = _resolve_input_type(item.get("input_type"))
            enabled = bool(item.get("enabled", True))
            reasoning_effort = _resolve_reasoning_effort(item.get("reasoning_effort"))

            agent = _create_agent(
                session=session,
                name=name,
                description=description,
                prompt=prompt,
                arguments=item.get("arguments", []),
                output_model=item.get("output_model"),
                router=router,
                agent_type=agent_type,
                input_type=input_type,
                enabled=enabled,
                reasoning_effort=reasoning_effort,
                parser_path=item.get("parser_path"),
                given=item.keys(),
            )
            created_agents[name] = agent

            # Tools
            tools_cfg = item.get("tools", {})
            tool_names = tools_cfg.get("names", []) or []
            tool_limits = tools_cfg.get("limits", None)
            tool_required = tools_cfg.get("required", None)
            if tool_names:
                _attach_tools(session, agent, tool_names, tool_limits, tool_required)
        # Sub-agent relationships
        for rel in subs_cfg:
            parent_name = rel["parent"]
            child_name = rel["child"]
            limit = rel.get("limit")

            parent = session.exec(
                select(Agent).where(Agent.name == parent_name)
            ).first()
            child = session.exec(select(Agent).where(Agent.name == child_name)).first()
            if not parent or not child:
                print(
                    f"[populate_agents] Warning: SubAgent relation '{parent_name}' -> '{child_name}' skipped (missing agent)."
                )
                continue

            _create_sub_agent(session, parent, child, limit)

        session.commit()
        print("[populate_agents] Agent population from JSON complete.")
