from typing import Literal

from pydantic import Field

from templates.common import TemplateModel


class UiExceptionReport(TemplateModel):
    result: str = Field(
        ...,
        description="Summary of the recovery execution result.",
    )
    success: bool = Field(
        ...,
        description="Indicates if the recovery was successful.",
    )
    process_finished: bool = Field(
        ...,
        description=(
            "Whether the recovery completed the rest of the process, as set with "
            "`set_continuation`."
        ),
    )
    continue_from_model_event_id: str | None = Field(
        ...,
        description=(
            "The model event id (a task id of `model`) the robot runs next, as set "
            "with `set_continuation`; null when the process is finished or the "
            "recovery failed."
        ),
    )
    loop_iteration: list[int] = Field(
        ...,
        description=(
            "0-based passes of the loops enclosing that task, outermost first, as set "
            "with `set_continuation`; [] when there is none."
        ),
    )


class RecoveryInference(TemplateModel):
    """What a UI recovery must achieve, inferred from the robot's report."""

    failure_kind: Literal["navigation", "data_entry", "extraction", "other"] = Field(
        ...,
        description="The kind of interaction that failed.",
    )
    failed_model_event_id: str | None = Field(
        ...,
        description=(
            "The task id of `model` the failed activity performed; null when it "
            "can't be told."
        ),
    )
    loop_passes: list[int] = Field(
        ...,
        description=(
            "0-based pass of each loop enclosing the failed activity, outermost "
            "first; [] when it is in no loop."
        ),
    )
    recovery_task: str = Field(
        ...,
        description="The instruction for the GUI agent: the state the application must reach.",
    )
    rationale: str = Field(
        ...,
        description="Why this task gets the process back on track.",
    )


class ContinuationReport(TemplateModel):
    """Whether a UI recovery succeeded, once `set_continuation` named the outcome."""

    success: bool = Field(
        ...,
        description=(
            "Whether the recovery succeeded; agrees with the `recovered` passed to "
            "`set_continuation`."
        ),
    )
    summary: str = Field(
        ...,
        description="What the recovery did and where the robot continues.",
    )


__all__ = [
    "ContinuationReport",
    "RecoveryInference",
    "UiExceptionReport",
]
