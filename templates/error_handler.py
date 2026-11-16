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


__all__ = [
    "UiExceptionReport",
]
