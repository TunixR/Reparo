"""The gateway eval reports the output model the Gateway Orchestrator is seeded with."""

from strands.tools.structured_output.structured_output_utils import (
    convert_pydantic_to_tool_spec,
)

from evals.agents import gateway_output_model


def test_the_gateway_eval_uses_the_seeded_gateway_decision() -> None:
    tool_spec = convert_pydantic_to_tool_spec(gateway_output_model())

    assert tool_spec["name"] == "GatewayDecision"
    assert tool_spec["inputSchema"]["json"]["required"] == ["ui_exception", "rationale"]
