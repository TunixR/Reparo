from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from templates.common import TemplateModel


def _normalize_optional_id(value: str | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value if value.strip() else None
    raise ValueError("optional id fields must be str, int, or null")


class RecoveryUiLogEntry(TemplateModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    activity_id: str | None
    activity_name: str | None
    event_id: str
    event_name: str | None
    model_event_id: str | None = None
    action_type: str
    application: str | None
    input: str | None
    ui_element_target: str | None
    ui_group: str | None = None
    timestamp: str
    previous_state: str | None
    current_state: str | None

    @field_validator("case_id", "event_id", mode="before")
    @classmethod
    def normalize_required_id(cls, value: str | int) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("id fields must be non-empty")
            return value
        raise ValueError("id fields must be str or int")

    @field_validator("activity_id", "model_event_id", mode="before")
    @classmethod
    def normalize_optional_id(cls, value: str | int | None) -> str | None:
        return _normalize_optional_id(value)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_iso8601(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                "timestamp must be a valid ISO-8601 datetime string"
            ) from exc
        return value


class RecoveryErroredAct(TemplateModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    activity_id: str | None
    activity_name: str | None
    event_id: str
    event_name: str | None
    model_event_id: str | None = None
    action_type: str
    application: str | None
    input: str | None
    error_code: str
    error_description: str

    @field_validator("case_id", "event_id", mode="before")
    @classmethod
    def normalize_required_id(cls, value: str | int) -> str:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("id fields must be non-empty")
            return value
        raise ValueError("id fields must be str or int")

    @field_validator("activity_id", "model_event_id", mode="before")
    @classmethod
    def normalize_optional_id(cls, value: str | int | None) -> str | None:
        return _normalize_optional_id(value)


class RecoveryPayload(TemplateModel):
    """The one message a robot sends to start a recovery.

    `model` is the process model; its task ids are model event ids. UI log
    entries and the errored activity name the task they performed through
    `model_event_id`, while their `event_id` only identifies the occurrence in
    this run.
    """

    model_config = ConfigDict(extra="forbid")

    task_name: str
    platform: str
    os: str
    variables: dict[str, Any] = Field(default_factory=dict)
    ui_log: list[RecoveryUiLogEntry]
    errored_act: RecoveryErroredAct
    model: str

    @field_validator("model")
    @classmethod
    def validate_model_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must be a non-empty string")
        return value


__all__ = [
    "RecoveryErroredAct",
    "RecoveryPayload",
    "RecoveryUiLogEntry",
]
