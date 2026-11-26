from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BARE_PAYLOAD_KEYS = ("task_name", "ui_log", "errored_act")


def _read_catalog(payload_file: Path) -> list[Any]:
    """The catalog's entries; a file holding one object is a catalog of one."""
    if not payload_file.exists():
        raise ValueError(f"Payload file not found: {payload_file}")

    try:
        raw = json.loads(payload_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in payload file: {payload_file}") from exc

    if isinstance(raw, dict):
        return [raw]
    if not isinstance(raw, list):
        raise ValueError("Payload catalog must be a JSON array")  # noqa: TRY004
    return raw


def _wrapped_entry(
    item: dict[str, Any], seen_ids: set[int]
) -> tuple[int, dict[str, Any]]:
    """The id and payload of an `{'id': int, 'payload': {...}}` entry."""
    item_id = item["id"]
    if not isinstance(item_id, int):
        raise ValueError("Payload entry 'id' must be an integer")  # noqa: TRY004
    if item_id in seen_ids:
        raise ValueError(f"Duplicate payload id found: {item_id}")
    seen_ids.add(item_id)

    payload = item["payload"]
    if not isinstance(payload, dict):
        raise ValueError("Payload entry 'payload' must be an object")  # noqa: TRY004
    return item_id, payload


def _catalog_entry(
    item: Any, position: int, seen_ids: set[int]
) -> tuple[int, dict[str, Any]]:
    """The id and payload of one entry. A bare payload's id is its 1-based position."""
    if not isinstance(item, dict):
        raise ValueError("Each payload catalog entry must be an object")  # noqa: TRY004
    if "id" in item and "payload" in item:
        return _wrapped_entry(item, seen_ids)
    if not all(key in item for key in BARE_PAYLOAD_KEYS):
        raise ValueError(
            "Each entry must be either {'id': int, 'payload': {...}} or a payload object"
        )
    return position, item


def load_payload_by_id(payload_file: Path, payload_id: int) -> dict[str, Any]:
    # Entries are checked up to the match; the ones after it are not read.
    seen_ids: set[int] = set()
    for position, item in enumerate(_read_catalog(payload_file), start=1):
        item_id, payload = _catalog_entry(item, position, seen_ids)
        if item_id == payload_id:
            return payload
    raise ValueError(f"Payload id not found: {payload_id}")
