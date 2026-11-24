"""How each agent type runs; the only place that branches on `AgentType`.

`behaviour_of(agent)` gives the class that runs an agent row, `validate_agent` runs
its type's rules, and `agent_types_of` lists the types of a kind, such as every GUI
agent type. Importing this package registers every type:

    AgentBehaviour              base.py: tool, input binding, traces, usage
    ├── DefaultAgent            default.py: AgentType.Agent
    └── GuiAgent (abstract)     gui/base.py: fixed inputs, setup, loop skeleton
        ├── TooledGuiAgent      gui/tooled.py: actions as Strands tools
        └── TextGuiAgent        gui/text.py: actions read by a parser
"""

from database.agents.behaviours.base import (
    AgentBehaviour,
    agent_types_of,
    behaviour_class,
    behaviour_of,
    validate_agent,
)
from database.agents.behaviours.default import DefaultAgent
from database.agents.behaviours.gui import GuiAgent, TextGuiAgent, TooledGuiAgent
from database.agents.models import Agent

__all__ = [
    "AgentBehaviour",
    "DefaultAgent",
    "GuiAgent",
    "TextGuiAgent",
    "TooledGuiAgent",
    "agent_types_of",
    "behaviour_class",
    "behaviour_of",
    "is_gui_agent",
    "validate_agent",
]


def is_gui_agent(agent: Agent) -> bool:
    """Whether ``agent`` drives the robot's desktop."""
    return issubclass(behaviour_class(agent.type), GuiAgent)
