from typing import Any

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from templates.common import TemplateModel


class ResponseToRPA(TemplateModel):
    """The recovery outcome sent to the robot; only `success` is reported by the agent."""

    success: bool = Field(
        ...,
        description="Indicates whether the recovery was successful and the RPA can continue.",
    )
    process_finished: SkipJsonSchema[bool] = Field(
        default=False,
        description="Whether the recovery completed the rest of the process.",
    )
    continue_from_model_event_id: SkipJsonSchema[str | None] = Field(
        default=None,
        description=(
            "The task id of `model` the robot runs next; null when the process is "
            "finished or the recovery failed."
        ),
    )
    loop_iteration: SkipJsonSchema[list[int]] = Field(
        default_factory=list,
        description=(
            "0-based pass of each loop enclosing that task, outermost first; [] when "
            "it is in no loop, the process is finished or the recovery failed."
        ),
    )
    variables: SkipJsonSchema[dict[str, Any]] = Field(
        default_factory=dict,
        description="The payload's variables as left by the recovery.",
    )


__all__ = [
    "ResponseToRPA",
]
