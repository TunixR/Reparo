"""A robot without vision can't be served by today's GUI agents."""

import pytest

from robot.capabilities import CAPABILITIES_STATE_KEY, require_vision


@pytest.mark.parametrize(
    "state",
    [
        {},
        {CAPABILITIES_STATE_KEY: None},
        {CAPABILITIES_STATE_KEY: {"modalities": ["dom", "vision"], "ocr": False}},
    ],
    ids=["undeclared", "none", "vision-not-first"],
)
def test_require_vision_accepts_robots_that_support_vision(state: dict[str, object]):
    require_vision(state)


def test_require_vision_rejects_robots_without_vision():
    state = {CAPABILITIES_STATE_KEY: {"modalities": ["dom", "a11y"], "ocr": True}}

    with pytest.raises(ValueError, match="only dom, a11y"):
        require_vision(state)
