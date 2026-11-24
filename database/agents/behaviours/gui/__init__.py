"""GUI agents: they drive the robot's desktop from screenshots (see `GuiAgent`)."""

from database.agents.behaviours.gui.base import GuiAgent
from database.agents.behaviours.gui.shared import (
    DEFAULT_GUI_ACTIONS,
    GUI_ARGUMENTS,
    GuiRunReport,
)
from database.agents.behaviours.gui.text import TextGuiAgent
from database.agents.behaviours.gui.tooled import TooledGuiAgent

__all__ = [
    "DEFAULT_GUI_ACTIONS",
    "GUI_ARGUMENTS",
    "GuiAgent",
    "GuiRunReport",
    "TextGuiAgent",
    "TooledGuiAgent",
]
