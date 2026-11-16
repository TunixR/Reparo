"""The seeded agents' prompts match the agents: their files, tools and placeholders."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from database.agents.tooled_gui_agent import TOOL_NAME as GUI_AGENT_TOOL_NAME
from database.agents.tooled_gui_agent import TooledGuiAgent
from database.populators.agents import DEFAULT_PROMPTS_DIR, _read_prompt  # pyright: ignore[reportPrivateUsage]
from database.populators.tools import _gather_tool_functions  # pyright: ignore[reportPrivateUsage]

CONFIG: dict[str, Any] = json.loads(
    (DEFAULT_PROMPTS_DIR.parent / "agents.json").read_text(encoding="utf-8")
)
AGENTS: dict[str, dict[str, Any]] = {a["name"]: a for a in CONFIG["agents"]}


def _sub_agent_tool_name(agent: dict[str, Any]) -> str:
    """The tool name an agent is exposed as (`Agent.get_tool_name`)."""
    if agent["agent_type"] == "GuiAgent":
        return GUI_AGENT_TOOL_NAME
    return agent["name"].lower().replace(" ", "_")


def _gui_tool_names() -> set[str]:
    router = SimpleNamespace(relative_dim=None, original_coords=None)
    gui = TooledGuiAgent(SimpleNamespace(router=router))  # pyright: ignore[reportArgumentType]
    return {t.tool_name for t in gui.get_gui_tools({})}


GUI_TOOLS = _gui_tool_names()
ALL_TOOLS = (
    {t.tool_name for t in _gather_tool_functions()}
    | {_sub_agent_tool_name(a) for a in AGENTS.values()}
    | GUI_TOOLS
)


def _given_tools(agent: dict[str, Any]) -> set[str]:
    """The tools an agent built from this seed gets, sub-agents included."""
    if agent["agent_type"] == "GuiAgent":
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


@pytest.mark.parametrize("name", ["Gateway Orchestrator", "UI Exception Handler"])
def test_prompt_names_every_tool_the_agent_is_given(name: str) -> None:
    agent = AGENTS[name]
    prompt = _read_prompt(name, agent["prompt_file"])

    assert _named_tools(prompt) == _given_tools(agent)


@pytest.mark.parametrize(
    "name", sorted(n for n, a in AGENTS.items() if a["agent_type"] == "GuiAgent")
)
def test_gui_prompt_takes_only_the_instruction(name: str) -> None:
    # GUI agents fill their prompt with str.format(instruction=...).
    prompt = _read_prompt(name, AGENTS[name]["prompt_file"])

    assert "<the instruction>" in prompt.format(instruction="<the instruction>")


def test_prompt_file_must_be_in_the_default_prompts_folder() -> None:
    with pytest.raises(ValueError, match="must be a file name"):
        _ = _read_prompt("Some Agent", "../agents.json")
