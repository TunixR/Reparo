"""Task functions that wire Reparo agents to Strands Evals.

Each function creates a real Strands Agent with the original system prompt,
invokes it with the test case input, and returns a dict that includes both
evaluation fields and full model responses for traceability.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

import regex
from strands import Agent as StrandsAgent
from strands.models.openai import OpenAIModel
from strands.types.content import ContentBlock
from strands_evals import Case
from tenacity import retry, stop_after_attempt

import settings
from database.agents.specs import OutputModelSpec, build_output_model
from database.populators.agents import agent_seed
from evals.prompts import GATEWAY_PROMPT, TASK_INFERENCE_ONLY
from templates.common import TemplateModel


def gateway_output_model() -> type[TemplateModel]:
    """The class the seeded Gateway Orchestrator reports, built from ``agents.json``."""
    seed = agent_seed("Gateway Orchestrator")
    if seed is None or seed.get("output_model") is None:
        raise LookupError("agents.json seeds no Gateway Orchestrator output model.")
    return build_output_model(OutputModelSpec.model_validate(seed["output_model"]))


def build_model(
    model_name: str | None = None,
    api_url: str | None = None,
) -> OpenAIModel:
    """Create an OpenAIModel from environment config or CLI overrides."""
    return OpenAIModel(
        client_args={
            "api_key": settings.PROVIDER_API_KEY or settings.FREE_PROVIDER_API_KEY,
            "base_url": api_url or settings.PROVIDER_API_BASE,
        },
        model_id=model_name or settings.PROVIDER_MODEL,
    )


def _extract_agent_messages(agent: StrandsAgent) -> list[dict[str, Any]]:
    """Extract agent messages as serializable dicts."""
    messages = []
    for msg in agent.messages:
        serializable: dict[str, Any] = {"role": msg.get("role", "unknown")}
        serializable["content"] = []
        # dont include bytes
        messages.append(serializable)
        content = msg.get("content", [])
        for block in content:
            if not ("image" in block or block.get("type", "") == "image"):
                messages[-1]["content"].append(block)
    return messages


def _extract_final_task(agent: StrandsAgent) -> str:
    """Extract the <Final task> content from the agent's last response."""
    messages = _extract_agent_messages(agent)
    if not messages:
        return ""

    last_msg = messages[-1]
    content = last_msg.get("content", [])
    if not content:
        return ""

    # Get the text from the last content block
    text = ""
    for block in reversed(content):
        if isinstance(block, dict) and "text" in block:
            text = block["text"]
            break
    if not text:
        return ""

    if match := regex.search(r'\{(?:[^{}"\\]+|"(?:\\.|[^"\\])*"|(?R))*\}', text):
        try:
            json_content = json.loads(match.group(0))
            if "final_task" in json_content:
                return json_content["final_task"].strip()
        except json.JSONDecodeError:
            pass  # If JSON parsing fails, continue to regex extraction

    # Fallback: try "final_task": ... (without closing tag)
    if match := re.search(r'"final_task":\s*(.*?)$', text, re.DOTALL):
        return match.group(1).strip()

    return "TASK_NOT_FOUND"


@retry(stop=stop_after_attempt(3), reraise=True)
def gateway_task(case: Case[str, str], mocks: Any) -> dict[str, Any]:
    """Invoke the Gateway Orchestrator for error filtering evaluation.

    As in production, the gateway runs freely, then reports its seeded output model.

    Returns a dict with:
    - is_ui_error: the decision's ``ui_exception`` (for evaluator)
    - model_messages: full conversation trace (for debuggability)
    - tool_calls: which mock tools were called and with what args
    - raw_instruction: what was sent to the model
    """
    mocks.clear()
    error_input = json.loads(case.input)
    model = build_model()

    agent = StrandsAgent(
        model=model,
        tools=mocks.build_tools(),
        system_prompt=GATEWAY_PROMPT,
        callback_handler=None,
    )

    instruction = (
        f"Error code: {error_input.get('code', 'UNKNOWN')}\n"
        f"Variables: {json.dumps(error_input.get('variables', {}))}\n"
        f"Details: {json.dumps(error_input.get('details', {}))}\n\n"
        f"Analyze this error and route it to the appropriate handler."
    )

    _ = agent(instruction)
    decision_model = gateway_output_model()
    decision = agent(
        "Given our conversation so far, please provide a report using structured "
        "output with the given tool.",
        structured_output_model=decision_model,
    ).structured_output
    if not isinstance(decision, decision_model):
        raise TypeError(f"The gateway did not report a {decision_model.__name__}.")
    reported = decision.model_dump()

    return {
        "is_ui_error": reported["ui_exception"],
        "rationale": reported["rationale"],
        "model_messages": _extract_agent_messages(agent),
        "tool_calls": {"route_to_human": mocks.human_calls},
        "raw_instruction": instruction,
    }


def _extract_agent_metrics(
    result: Any,
    first_token_time: float | None = None,
    last_token_time: float | None = None,
) -> dict[str, Any]:
    """Extract token usage and timing metrics from an AgentResult.

    Args:
        result: The AgentResult from the agent invocation.
        first_token_time: Wall-clock time when the first token was received.
        last_token_time: Wall-clock time when the last token was received.

    Returns a dict with:
    - input_tokens, output_tokens, total_tokens
    - total_duration_s: wall-clock time across all cycles
    - first_to_last_token_s: time from first token to last token
    """
    usage = result.metrics.accumulated_usage

    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    total_tokens = usage.get("totalTokens", 0)

    total_duration_s = sum(result.metrics.cycle_durations)

    first_to_last_token_s = 0.0
    if first_token_time and last_token_time:
        first_to_last_token_s = last_token_time - first_token_time

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_duration_s": round(total_duration_s, 3),
        "first_to_last_token_s": round(first_to_last_token_s, 3),
    }


def _load_image_content(image_path: Path) -> ContentBlock:
    """Load an image file and return an ImageContent dict for Strands."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    suffix = image_path.suffix.lower().lstrip(".")
    format_map = {
        "png": "png",
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "gif": "gif",
        "webp": "webp",
    }
    img_format: Literal["png", "jpeg", "gif", "webp"] = format_map.get(suffix, "png")  # pyright: ignore[reportAssignmentType]

    return {
        "image": {
            "format": img_format,
            "source": {"bytes": image_bytes},
        }
    }


def _handler_instruction(payload: dict[str, Any]) -> str:
    """The text instruction describing a recovery payload to the task inference agent."""
    return (
        f"task_name: {payload.get('task_name', '')},\n"
        f"platform={payload.get('platform', '')},\n"
        f"os={payload.get('os', '')},\n"
        f"variables={json.dumps(payload.get('variables', {}))},\n"
        f"ui_log={json.dumps(payload.get('ui_log', []))},\n"
        f"errored_act={json.dumps(payload.get('errored_act', {}))},\n"
        f"model={payload.get('model', '')}"
    )


def _handler_prompt(
    instruction: str, image_path: Path | None
) -> str | list[ContentBlock]:
    """The instruction alone, or with the screenshot at ``image_path`` when it exists."""
    if image_path is None or not image_path.exists():
        return instruction
    return [ContentBlock(text=instruction), _load_image_content(image_path)]


async def _stream_with_token_times(
    agent: StrandsAgent, prompt: str | list[ContentBlock]
) -> tuple[Any, float | None, float | None]:
    """Stream the agent's answer to ``prompt``.

    Returns (agent_result, first_token_time, last_token_time): the final event's
    AgentResult (with metrics) and the wall-clock times of the first and last
    content events, which carry the token data.
    """
    first_token_time: float | None = None
    last_token_time: float | None = None
    agent_result: Any = None

    async for event in agent.stream_async(prompt):
        now = time.time()
        if not isinstance(event, dict):
            continue
        if "data" in event:
            if first_token_time is None:
                first_token_time = now
            last_token_time = now
        if "result" in event:
            agent_result = event["result"]

    return agent_result, first_token_time, last_token_time


@retry(stop=stop_after_attempt(3), reraise=True)
def _run_handler(
    case: Case[str, str],
    *,
    api_url: str | None = None,
    model_name: str | None = None,
    image_path: Path | None = None,
) -> tuple[dict[str, Any], Any, float | None, float | None]:
    """Run the handler agent, tracking first/last token times.

    Uses stream_async to capture precise timing and the AgentResult (with metrics)
    from a single invocation.

    Returns (task_output, agent_result, first_token_time, last_token_time).
    """
    payload = json.loads(case.input)
    agent = StrandsAgent(
        model=build_model(model_name=model_name, api_url=api_url),
        system_prompt=TASK_INFERENCE_ONLY,
        callback_handler=None,
    )
    instruction = _handler_instruction(payload)
    prompt = _handler_prompt(instruction, image_path)

    agent_result, first_token_time, last_token_time = asyncio.run(
        _stream_with_token_times(agent, prompt)
    )

    return (
        {
            "generated_task": _extract_final_task(agent),
            "model_messages": _extract_agent_messages(agent),
            "raw_instruction": instruction,
        },
        agent_result,
        first_token_time,
        last_token_time,
    )


def handler_task(
    case: Case[str, str],
    *,
    api_url: str | None = None,
    model_name: str | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:
    """Invoke the agent for task identification evaluation.

    Uses stream_async for precise first-to-last token timing and metrics
    from a single invocation. No mock tools — the agent generates the
    recovery task directly.

    Returns a dict with:
    - generated_task: str (the extracted recovery task)
    - model_messages: full conversation trace
    - raw_instruction: what was sent to the model
    - agent_metrics: token usage and timing metrics
    """
    task_output, agent_result, first_token_time, last_token_time = _run_handler(
        case,
        api_url=api_url,
        model_name=model_name,
        image_path=image_path,
    )

    if agent_result is not None:
        task_output["agent_metrics"] = _extract_agent_metrics(
            agent_result, first_token_time, last_token_time
        )
    else:
        # Fallback: no result captured (shouldn't happen)
        task_output["agent_metrics"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_duration_s": 0.0,  # nosec
            "first_to_last_token_s": 0.0,  # nosec
        }

    return task_output
