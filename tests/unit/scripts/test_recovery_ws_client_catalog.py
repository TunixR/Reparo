"""How the reference client picks a payload out of its catalog file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.recovery_ws_client.payloads import load_payload_by_id

BARE = {"task_name": "Approve", "ui_log": [], "errored_act": "click"}


def _catalog(tmp_path: Path, content: Any) -> Path:
    path = tmp_path / "payloads.json"
    _ = path.write_text(json.dumps(content), encoding="utf-8")
    return path


def test_a_wrapped_entry_is_found_by_its_id(tmp_path: Path) -> None:
    path = _catalog(
        tmp_path, [{"id": 7, "payload": {"a": 1}}, {"id": 3, "payload": {"b": 2}}]
    )

    assert load_payload_by_id(path, 3) == {"b": 2}


def test_a_bare_entry_is_found_by_its_position(tmp_path: Path) -> None:
    path = _catalog(tmp_path, [{"id": 9, "payload": {"a": 1}}, BARE])

    assert load_payload_by_id(path, 2) == BARE


def test_a_single_object_is_a_catalog_of_one(tmp_path: Path) -> None:
    path = _catalog(tmp_path, BARE)

    assert load_payload_by_id(path, 1) == BARE


def test_entries_after_the_match_are_not_checked(tmp_path: Path) -> None:
    path = _catalog(tmp_path, [{"id": 1, "payload": {"a": 1}}, "junk"])

    assert load_payload_by_id(path, 1) == {"a": 1}


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Payload file not found"):
        _ = load_payload_by_id(tmp_path / "nope.json", 1)


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "payloads.json"
    _ = path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON") as excinfo:
        _ = load_payload_by_id(path, 1)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("text", "Payload catalog must be a JSON array"),
        (["text"], "Each payload catalog entry must be an object"),
        ([{"id": 1}], "Each entry must be either"),
        ([{"id": "1", "payload": {}}], "Payload entry 'id' must be an integer"),
        (
            [{"id": 1, "payload": {}}, {"id": 1, "payload": {}}],
            "Duplicate payload id found: 1",
        ),
        ([{"id": 1, "payload": []}], "Payload entry 'payload' must be an object"),
        ([{"id": 1, "payload": {}}, BARE], "Payload id not found: 5"),
    ],
)
def test_a_bad_catalog_is_reported(tmp_path: Path, content: Any, error: str) -> None:
    path = _catalog(tmp_path, content)

    with pytest.raises(ValueError, match=error):
        _ = load_payload_by_id(path, 5)
