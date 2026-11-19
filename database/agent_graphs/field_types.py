"""The types of fields in node outputs and recovery inputs, for validating graphs.

Paths are dotted field names into a pydantic model, as conditions and input maps
write them. A path may go through nested models and dicts; past an ``Any`` its
type is unknown, so anything goes.
"""

from __future__ import annotations

import json
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from database.agent_graphs.conditions import Operator

_ORDERING: frozenset[Operator] = frozenset({"lt", "le", "gt", "ge"})
_NULL_CHECKS: frozenset[Operator] = frozenset({"is_null", "not_null"})
_MEMBERSHIP: frozenset[Operator] = frozenset({"in", "not_in"})


class FieldPathError(ValueError):
    """A path that doesn't lead to a field; the message says where it breaks."""


def field_type(model: type[BaseModel], path: str) -> Any:
    """The annotation of the field at the dotted ``path`` in ``model``."""
    annotation: Any = model
    walked: list[str] = []
    for name in path.split("."):
        inner = _without_none(annotation)
        if inner is Any:
            return Any
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            field = inner.model_fields.get(name)
            if field is None:
                raise FieldPathError(
                    f"'{name}' isn't a field of {inner.__name__} "
                    f"(fields: {_field_list(inner)})"
                )
            annotation = field.annotation
        elif inner is dict or get_origin(inner) is dict:
            annotation = (get_args(inner) or (Any, Any))[1]
        else:
            raise FieldPathError(f"'{'.'.join(walked)}' has no fields")
        walked.append(name)
    return annotation


def comparison_error(annotation: Any, op: Operator, value: JsonValue) -> str | None:
    """Why ``op`` with ``value`` doesn't fit a field typed ``annotation``, if it doesn't."""
    if annotation is Any:
        return None
    if op in _NULL_CHECKS:
        if _nullable(annotation):
            return None
        return f"it's never null, so '{op}' always gives the same answer"
    if op in _ORDERING:
        annotation = _without_none(annotation)
        if not _orderable(annotation):
            return f"'{op}' can't order a {_name(annotation)}; order numbers or strings"
    items: list[JsonValue] = (
        value if op in _MEMBERSHIP and isinstance(value, list) else [value]
    )
    adapter: TypeAdapter[Any] = TypeAdapter(annotation)
    for item in items:
        try:
            _ = adapter.validate_json(json.dumps(item), strict=True)
        except ValidationError:
            return f"{json.dumps(item)} isn't a valid {_name(annotation)}"
    return None


def _nullable(annotation: Any) -> bool:
    return annotation is Any or _split_none(annotation)[1]


def _orderable(annotation: Any) -> bool:
    """Whether the JSON values of the type compare with ``<``: numbers or strings."""
    if get_origin(annotation) is Literal:
        return all(_orderable(type(arg)) for arg in get_args(annotation))
    return (
        isinstance(annotation, type)
        and issubclass(annotation, (int, float, str))
        and not issubclass(annotation, bool)
    )


def _name(annotation: Any) -> str:
    return getattr(annotation, "__name__", None) or str(annotation)


def _without_none(annotation: Any) -> Any:
    """The annotation without ``None``, if that leaves a single type."""
    rest, _ = _split_none(annotation)
    return rest[0] if len(rest) == 1 else annotation


def _split_none(annotation: Any) -> tuple[list[Any], bool]:
    """The types a union of ``annotation`` allows besides ``None``, and whether it
    allows ``None``."""
    members = (
        list(get_args(annotation))
        if get_origin(annotation) in (Union, types.UnionType)
        else [annotation]
    )
    rest = [m for m in members if m is not None and m is not type(None)]
    return rest, len(rest) < len(members)


def _field_list(model: type[BaseModel]) -> str:
    return ", ".join(model.model_fields) or "none"
