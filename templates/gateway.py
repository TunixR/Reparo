from typing import Any

from pydantic import Field

from templates.common import TemplateModel


class ResponseToRPA(TemplateModel):
    success: bool = Field(
        ...,
        description="Indicates whether the recovery was successful and the RPA can continue.",
    )
    process_finished: bool = Field(
        ...,
        description=(
            "Whether the recovery completed the rest of the process. Pass through the "
            "value reported by the recovery module."
        ),
    )
    continue_from_model_event_id: str | None = Field(
        ...,
        description=(
            "The model event id (a task id of `model`) the RPA continues from. Pass "
            "through the value reported by the recovery module; null when the process "
            "is finished or the recovery failed."
        ),
    )
    loop_iteration: list[int] = Field(
        ...,
        description=(
            "0-based passes of the loops enclosing that task, outermost first. Pass "
            "through the value reported by the recovery module; [] when there is none."
        ),
    )


class RecoveryAnswer(TemplateModel):
    """The ``done.content`` sent to the robot.

    The continuation fields come from the continuation validated by
    ``set_continuation``, never from an LLM's structured output, and ``variables``
    are the payload's variables as left by the recovery.
    """

    success: bool = Field(
        ..., description="Whether the recovery succeeded and the robot can continue."
    )
    process_finished: bool = Field(
        ..., description="Whether the recovery completed the rest of the process."
    )
    continue_from_model_event_id: str | None = Field(
        ...,
        description=(
            "The task id of `model` the robot runs next; null when the process is "
            "finished or the recovery failed."
        ),
    )
    loop_iteration: list[int] = Field(
        ...,
        description=(
            "0-based pass of each loop enclosing that task, outermost first; [] when "
            "it is in no loop, the process is finished or the recovery failed."
        ),
    )
    variables: dict[str, Any] = Field(
        ..., description="The payload's variables as left by the recovery."
    )


__all__ = [
    "RecoveryAnswer",
    "ResponseToRPA",
]
