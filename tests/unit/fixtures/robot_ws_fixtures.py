"""Fixtures for tests that talk to the robot WebSocket connections."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

import routers.evals as evals_module
from database.agents.models import Agent, AgentType
from database.provider.models import Router


@pytest.fixture
def eval_gui_agent(
    make_agent: Callable[..., Agent],
    mock_router: Router,
    monkeypatch: pytest.MonkeyPatch,
) -> Agent:
    """A GUI agent that ``WS /evals/recovery/ws`` resolves whatever its ``X-AGENT-ID``."""
    agent = make_agent(
        router=mock_router,
        name="GUI",
        description="GUI agent",
        prompt="prompt",
        type=AgentType.TooledGuiAgent,
    )
    # The agent lookup isn't under test (it passes the raw header to a UUID column).
    monkeypatch.setattr(evals_module, "_get_valid_agent", lambda *_: agent)
    return agent


def receive_skipping_pings(ws: Any) -> dict[str, Any]:
    """The next JSON message from ``ws`` that isn't a keep-alive ping."""
    while True:
        message = ws.receive_json()
        if message.get("type") != "ping":
            return message
