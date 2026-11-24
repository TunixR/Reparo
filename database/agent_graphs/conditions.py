"""Edge conditions: declarative JSON predicates over a node's structured output.

A condition is validated into these models wherever it enters, stored as JSON by
`ConditionType`, and evaluated by walking it; it is never ``eval``'d.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)
from sqlalchemy import Dialect
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

type Operator = Literal[
    "eq", "ne", "in", "not_in", "lt", "le", "gt", "ge", "is_null", "not_null"
]


def _equal(actual: Any, expected: Any) -> bool:
    """JSON equality: unlike in Python, ``true`` doesn't equal ``1``."""
    return isinstance(actual, bool) == isinstance(expected, bool) and actual == expected


_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": _equal,
    "ne": lambda actual, expected: not _equal(actual, expected),
    "in": lambda actual, expected: any(_equal(actual, item) for item in expected),
    "not_in": lambda actual, expected: (
        not any(_equal(actual, item) for item in expected)
    ),
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "is_null": lambda actual, _: actual is None,
    "not_null": lambda actual, _: actual is not None,
}


class Compare(BaseModel):
    """Compares the value at a dotted ``field`` path with ``value``."""

    model_config = ConfigDict(extra="forbid")

    field: str
    op: Operator
    value: JsonValue = None

    @model_validator(mode="after")
    def _membership_takes_a_list(self) -> Compare:
        if self.op in ("in", "not_in") and not isinstance(self.value, list):
            raise ValueError(f"'{self.op}' compares against a list of values.")
        return self

    def holds(self, output: Mapping[str, Any]) -> bool:
        actual: Any = output
        for name in self.field.split("."):
            actual = actual[name]
        return bool(_OPERATORS[self.op](actual, self.value))


class AllOf(BaseModel):
    """Holds when every condition does, so an empty list holds."""

    model_config = ConfigDict(extra="forbid")

    all: list[Condition]

    def holds(self, output: Mapping[str, Any]) -> bool:
        return all(condition.holds(output) for condition in self.all)


class AnyOf(BaseModel):
    """Holds when some condition does, so an empty list doesn't."""

    model_config = ConfigDict(extra="forbid")

    any: list[Condition]

    def holds(self, output: Mapping[str, Any]) -> bool:
        return any(condition.holds(output) for condition in self.any)


class Not(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    not_: Condition = Field(alias="not")

    def holds(self, output: Mapping[str, Any]) -> bool:
        return not self.not_.holds(output)


type Condition = Compare | AllOf | AnyOf | Not

CONDITION: TypeAdapter[Condition] = TypeAdapter(Condition)


class ConditionType(TypeDecorator[Condition]):
    """A JSONB column that only stores, and always loads, a valid `Condition`."""

    impl = JSONB
    cache_ok = True

    def process_bind_param(
        self, value: Condition | Mapping[str, Any] | None, dialect: Dialect
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return CONDITION.dump_python(
            CONDITION.validate_python(value), mode="json", by_alias=True
        )

    def process_result_value(
        self, value: dict[str, Any] | None, dialect: Dialect
    ) -> Condition | None:
        return None if value is None else CONDITION.validate_python(value)
