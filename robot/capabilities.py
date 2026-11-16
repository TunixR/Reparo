"""What a robot declared it can be observed and driven through."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from templates.recovery_payload import RecoveryCapabilities

CAPABILITIES_STATE_KEY = "capabilities"
"""Invocation state key holding the robot's capabilities, as stored on its recovery."""


def require_vision(invocation_state: Mapping[str, Any]) -> None:
    """Fail unless the robot can be driven through screenshots.

    GUI agents only implement the ``vision`` modality for now, so a robot whose
    capabilities leave it out can't be served by any of them. A state without
    capabilities means the robot declared none, which defaults to ``vision``.

    Raises:
        ValueError: If the robot's capabilities don't include ``vision``.
    """
    raw = invocation_state.get(CAPABILITIES_STATE_KEY)
    capabilities = (
        RecoveryCapabilities()
        if raw is None
        else RecoveryCapabilities.model_validate(raw)
    )
    if "vision" not in capabilities.modalities:
        modalities = ", ".join(capabilities.modalities)
        raise ValueError(
            "This robot can't be driven through screenshots: its capabilities "
            f"list only {modalities}, and GUI agents support only vision. "
            "Don't retry a GUI agent for this recovery."
        )
