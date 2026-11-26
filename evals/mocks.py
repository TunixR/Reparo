"""Mock Strands @tool functions for evaluation capture.

Each mock captures the agent's tool calls so evaluators can inspect
what routing/identification decisions the agent made, without executing
real recovery logic.
"""

from __future__ import annotations

from typing import Any

from strands import tool


class GatewayMocks:
    """Mock tools for Gateway Orchestrator (Contribution 1: Error Filtering).

    Provides:
    - route_to_human: mock that captures when gateway escalates to human

    Whether the gateway classified a UI exception comes from its report.
    """

    def __init__(self) -> None:
        self.human_calls: list[dict[str, Any]] = []
        self._build_tools()

    def _build_tools(self) -> None:
        capture = self

        @tool(
            name="route_to_human",
            description="Route the error to a human operator for manual intervention.",
        )
        async def mock_route_human(error_data: str = "") -> dict[str, Any]:
            capture.human_calls.append({"error_data": error_data})
            return {"routed": True, "message": "Error escalated to human operator."}

        self._route_human = mock_route_human

    def build_tools(self) -> list[Any]:
        return [self._route_human]

    def clear(self) -> None:
        self.human_calls.clear()
