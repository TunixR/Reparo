"""The seeded agents' prompts match the agents: their files, tools and placeholders."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from database.agents.behaviours import GuiAgent, TextGuiAgent, behaviour_class
from database.agents.behaviours.gui.shared import DEFAULT_GUI_ACTIONS
from database.populators.agents import DEFAULT_PROMPTS_DIR, _read_prompt  # pyright: ignore[reportPrivateUsage]
from database.populators.tools import _gather_tool_functions  # pyright: ignore[reportPrivateUsage]

CONFIG: dict[str, Any] = json.loads(
    (DEFAULT_PROMPTS_DIR.parent / "agents.json").read_text(encoding="utf-8")
)
AGENTS: dict[str, dict[str, Any]] = {a["name"]: a for a in CONFIG["agents"]}


def _sub_agent_tool_name(agent: dict[str, Any]) -> str:
    """The tool name an agent is exposed as (`AgentBehaviour.tool_name`)."""
    return agent["name"].lower().replace(" ", "_")


def _is_gui(agent: dict[str, Any]) -> bool:
    return issubclass(behaviour_class(agent["agent_type"]), GuiAgent)


GUI_TOOLS = set(DEFAULT_GUI_ACTIONS.names())
ALL_TOOLS = (
    {t.tool_name for t in _gather_tool_functions()}
    | {_sub_agent_tool_name(a) for a in AGENTS.values()}
    | GUI_TOOLS
)


def _given_tools(agent: dict[str, Any]) -> set[str]:
    """The tools an agent built from this seed gets, sub-agents included."""
    if _is_gui(agent):
        return GUI_TOOLS
    given = set(agent.get("tools", {}).get("names", []))
    for link in CONFIG["sub_agents"]:
        child = AGENTS[link["child"]]
        if link["parent"] == agent["name"] and child.get("enabled", True):
            given.add(_sub_agent_tool_name(child))
    return given


def _named_tools(prompt: str) -> set[str]:
    """Framework tool names a prompt mentions in backticks (`name` or `name(...)`)."""
    return set(re.findall(r"`([a-z_]+)[`(]", prompt)) & ALL_TOOLS


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_prompt_names_only_tools_the_agent_is_given(name: str) -> None:
    agent = AGENTS[name]
    prompt = _read_prompt(name, agent["prompt_file"])

    assert _named_tools(prompt) <= _given_tools(agent)


@pytest.mark.parametrize(
    "name", ["Gateway Orchestrator", "UI Exception Handler", "Continuation Inferer"]
)
def test_prompt_names_every_tool_the_agent_is_given(name: str) -> None:
    agent = AGENTS[name]
    prompt = _read_prompt(name, agent["prompt_file"])

    assert _named_tools(prompt) == _given_tools(agent)


@pytest.mark.parametrize("name", sorted(n for n, a in AGENTS.items() if _is_gui(a)))
def test_gui_prompt_takes_only_its_placeholders(name: str) -> None:
    # A GUI agent fills its prompt with str.format(instruction=...); a TextGuiAgent
    # also gives it the actions in its parser's syntax.
    agent = AGENTS[name]
    prompt = _read_prompt(name, agent["prompt_file"])
    placeholders = {"instruction": "<the instruction>"}
    if behaviour_class(agent["agent_type"]) is TextGuiAgent:
        placeholders["actions"] = "<the actions>"

    filled = prompt.format(**placeholders)

    assert all(value in filled for value in placeholders.values())


def test_prompt_file_must_be_in_the_default_prompts_folder() -> None:
    with pytest.raises(ValueError, match="must be a file name"):
        _ = _read_prompt("Some Agent", "../agents.json")
