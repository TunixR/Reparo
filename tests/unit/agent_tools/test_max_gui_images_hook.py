from copy import deepcopy
from typing import Any

import pytest
from strands.hooks import BeforeModelCallEvent

from agent_tools.hooks import MaxGuiImagesHook
from tests.unit.agent_tools._shared.fakes import FakeRegistry, is_bound_method_of

PLACEHOLDER = {"text": "screenshot omitted to save context"}


def _image() -> dict[str, Any]:
    return {"image": {"format": "jpeg", "source": {"bytes": b"screenshot"}}}


class _FakeAgent:
    def __init__(self, messages: list[dict[str, Any]]):
        self.messages = messages


class _FakeBeforeModelCallEvent:
    def __init__(self, messages: list[dict[str, Any]]):
        self.agent = _FakeAgent(messages)


def _run_hook(hook: MaxGuiImagesHook, messages: list[dict[str, Any]]) -> None:
    hook.limit_images(_FakeBeforeModelCallEvent(messages))  # pyright: ignore[reportArgumentType]


def test_register_hooks_adds_model_call_callback():
    registry = FakeRegistry()
    hook = MaxGuiImagesHook(max_images=2)

    hook.register_hooks(registry)  # pyright: ignore[reportArgumentType]

    assert len(registry.callbacks) == 1
    event_type, callback = registry.callbacks[0]
    assert event_type is BeforeModelCallEvent
    assert is_bound_method_of(hook, callback)


@pytest.mark.parametrize(
    "kwargs", [{"max_images": -1}, {"max_images": 2, "prune_chunk": -1}]
)
def test_negative_limits_raise(kwargs: dict[str, int]):
    with pytest.raises(ValueError):
        _ = MaxGuiImagesHook(**kwargs)


def test_zero_chunk_keeps_only_the_last_images_across_messages():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": "start"}, _image()]},
        {"role": "assistant", "content": [{"text": "ok"}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "1",
                        "status": "success",
                        "content": [_image(), _image()],
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "2",
                        "status": "success",
                        "content": [_image()],
                    }
                }
            ],
        },
    ]

    _run_hook(MaxGuiImagesHook(max_images=2, prune_chunk=0), messages)

    assert messages[0]["content"][1] == PLACEHOLDER
    assert messages[2]["content"][0]["toolResult"]["content"][0] == PLACEHOLDER
    assert "image" in messages[2]["content"][0]["toolResult"]["content"][1]
    assert "image" in messages[3]["content"][0]["toolResult"]["content"][0]


def test_keeps_images_untouched_when_under_limit():
    first_image = _image()
    second_image = _image()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [first_image]},
        {"role": "user", "content": [{"toolResult": {"content": [second_image]}}]},
    ]

    _run_hook(MaxGuiImagesHook(max_images=2, prune_chunk=0), messages)

    assert messages[0]["content"][0] is first_image
    assert messages[1]["content"][0]["toolResult"]["content"][0] is second_image


def test_zero_max_images_and_chunk_replace_all_images():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [_image(), {"text": "keep me"}]},
        {
            "role": "user",
            "content": [{"toolResult": {"content": [_image(), {"text": "also keep"}]}}],
        },
    ]

    _run_hook(MaxGuiImagesHook(max_images=0, prune_chunk=0), messages)

    assert messages[0]["content"][0] == PLACEHOLDER
    assert messages[0]["content"][1] == {"text": "keep me"}
    assert messages[1]["content"][0]["toolResult"]["content"][0] == PLACEHOLDER
    assert messages[1]["content"][0]["toolResult"]["content"][1] == {
        "text": "also keep"
    }


def test_text_only_messages_are_untouched():
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "assistant", "content": [{"text": "hi"}]},
    ]

    _run_hook(MaxGuiImagesHook(max_images=1, prune_chunk=0), messages)

    assert messages == [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "assistant", "content": [{"text": "hi"}]},
    ]


def _step(messages: list[dict[str, Any]]) -> None:
    """Add the two messages of one GUI step: the action call and its screenshot."""
    tool_use_id = str(len(messages))
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": tool_use_id, "name": "click", "input": {}}}
            ],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [_image()],
                    }
                }
            ],
        }
    )


def _live_images(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        for block in message["content"]:
            if "image" in block:
                count += 1
            for result_block in block.get("toolResult", {}).get("content", []):
                if "image" in result_block:
                    count += 1
    return count


def _task_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"text": "task"}, _image()]}]


@pytest.mark.parametrize(
    ("max_images", "prune_chunk", "expected"),
    [
        (2, 3, [1, 2, 3, 4, 5, 2, 3, 4, 5, 2]),
        (2, 0, [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
        (0, 2, [1, 2, 0, 1, 2, 0, 1, 2, 0, 1]),
    ],
    ids=["batched", "zero-chunk-prunes-every-step", "zero-keep-cuts-to-none"],
)
def test_images_grow_to_keep_plus_chunk_before_cut_back_to_keep(
    max_images: int, prune_chunk: int, expected: list[int]
):
    hook = MaxGuiImagesHook(max_images=max_images, prune_chunk=prune_chunk)
    messages = _task_messages()

    live_after_each_call: list[int] = []
    for _ in range(10):
        _run_hook(hook, messages)
        live_after_each_call.append(_live_images(messages))
        _step(messages)

    assert live_after_each_call == expected


def test_history_is_unchanged_between_prunes():
    hook = MaxGuiImagesHook(max_images=2, prune_chunk=3)
    messages = _task_messages()
    for _ in range(5):
        _step(messages)
    _run_hook(hook, messages)  # 6 images > 2 + 3: prunes back to 2

    for _ in range(3):
        _step(messages)
        before = deepcopy(messages)
        _run_hook(hook, messages)
        assert messages == before
