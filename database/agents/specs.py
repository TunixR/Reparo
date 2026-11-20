"""Agent arguments and output models, defined as data.

A `FieldSpec` names one field and gives its type as a small type expression, such as
``"str | None"`` or ``"Literal['a', 'b']"``. The expression is parsed with `ast` and
walked against a whitelist; it is never evaluated. An agent's ``arguments`` are a list
of field specs; its ``output_model`` is an `OutputModelSpec`, whose name and description
are what the LLM sees as the structured-output tool.

These models hold every rule about a definition. The API types its payloads with them
and the `Agent` row validates its columns through them, so a definition is checked the
same way wherever it comes from. Pydantic classes built from a definition are cached
by its normalized form.
"""

import ast
import json
import keyword
from collections.abc import Sequence
from functools import cache
from types import NoneType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    MISSING,
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from templates.common import TemplateModel

_SCALARS: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "Any": Any,
    "list": list[Any],
    "dict": dict[str, Any],
}
_ALLOWED = (
    "str, int, float, bool, None, Any, list[T], dict[str, T], T | U, Literal[...]"
)
_NOT_SUPPORTED = {
    "Optional": "write 'T | None' instead of Optional[T]",
    "Union": "write 'T | U' instead of Union[T, U]",
    "tuple": "tuples are not supported, use list[T]",
    "set": "sets are not supported, use list[T]",
    "frozenset": "sets are not supported, use list[T]",
}


def parse_type(expression: str) -> Any:
    """Return the Python type a field's type expression stands for.

    Raises `ValueError` for anything outside the whitelist in `_ALLOWED`.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"{expression!r} is not a type expression; allowed: {_ALLOWED}"
        ) from exc
    return _type_of(tree.body)


def normalize_type(expression: str) -> str:
    """Return a type expression in one canonical spelling, e.g. ``str|None`` → ``str | None``."""
    return ast.unparse(ast.parse(expression.strip(), mode="eval"))


def _type_of(node: ast.expr) -> Any:
    match node:
        case ast.Name(id=name) if name in _SCALARS:
            return _SCALARS[name]
        case ast.Constant(value=None):
            return NoneType
        case ast.Subscript(value=ast.Name(id="list"), slice=item):
            if isinstance(item, ast.Tuple):
                raise ValueError(f"list takes one item type: list[T], not {_src(node)}")  # noqa: TRY004 - pydantic reports ValueError
            return list[_type_of(item)]
        case ast.Subscript(
            value=ast.Name(id="dict"), slice=ast.Tuple(elts=[ast.Name(id="str"), value])
        ):
            return dict[str, _type_of(value)]
        case ast.Subscript(value=ast.Name(id="dict")):
            raise ValueError(f"dict keys must be str: dict[str, T], not {_src(node)}")
        case ast.Subscript(value=ast.Name(id="Literal"), slice=values):
            items = values.elts if isinstance(values, ast.Tuple) else [values]
            if not items:
                raise ValueError("Literal needs at least one value")
            return Literal[tuple(_literal_value(item) for item in items)]  # pyright: ignore[reportInvalidTypeArguments] - built at runtime
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            return _type_of(left) | _type_of(right)
        case ast.Subscript(value=ast.Name(id="Optional" | "Union" as name), slice=args):
            members = args.elts if isinstance(args, ast.Tuple) else [args]
            if name == "Optional":
                members = [*members, ast.Constant(value=None)]
            union = " | ".join(ast.unparse(member) for member in members)
            raise ValueError(f"{_src(node)} is not supported: write {union!r}")
        case ast.Name(id=name) | ast.Subscript(value=ast.Name(id=name)) if (
            name in _NOT_SUPPORTED
        ):
            raise ValueError(f"{_src(node)} is not supported: {_NOT_SUPPORTED[name]}")
        case ast.Name(id=name):
            raise ValueError(
                f"unknown type {name!r} (nested models are not supported); "
                f"allowed: {_ALLOWED}"
            )
        case _:
            raise ValueError(
                f"{_src(node)} is not allowed in a type; allowed: {_ALLOWED}"
            )


def _literal_value(node: ast.expr) -> str | int | float | bool | None:
    """Return a Literal member, which must be a JSON scalar."""
    match node:
        case ast.Constant(value=str() | int() | float() | bool() | None as value) if (
            not isinstance(value, complex)
        ):
            return value
        case ast.UnaryOp(
            op=ast.USub(), operand=ast.Constant(value=int() | float() as value)
        ) if not isinstance(value, bool):
            return -value
        case _:
            raise ValueError(
                f"Literal values must be strings, numbers, booleans or None, not {_src(node)}"
            )


def _src(node: ast.expr) -> str:
    return repr(ast.unparse(node))


def _check_name(name: str) -> str:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{name!r} is not a Python identifier")
    return name


class FieldSpec(BaseModel):
    """One field of an agent's arguments or output model.

    Without a ``default`` the field is required, even when its type allows None; with
    one it is optional and the default must be a value of its type.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The field name, a Python identifier.")
    type: str = Field(
        description=(
            "The field type: str, int, float, bool, None, Any, list[T], dict[str, T], "
            "T | U, or Literal[...] over strings, numbers, booleans and None."
        ),
        examples=["str", "list[int]", "str | None", "Literal['a', 'b']"],
    )
    description: str = Field(description="What the field holds, shown to the LLM.")
    default: JsonValue | MISSING = Field(
        default=MISSING,
        description="The value when the field is left out; omit it to require the field.",
    )

    @field_validator("name")
    @classmethod
    def _valid_name(cls, name: str) -> str:
        _ = _check_name(name)
        if name.startswith(("_", "model_")):
            raise ValueError(f"field {name!r} must not start with '_' or 'model_'")
        return name

    @model_validator(mode="after")
    def _valid_type_and_default(self) -> Self:
        try:
            annotation = parse_type(self.type)
        except ValueError as exc:
            raise ValueError(f"field {self.name!r}: type {self.type!r}: {exc}") from exc
        self.type = normalize_type(self.type)
        if self.has_default:
            try:
                _ = TypeAdapter(
                    annotation, config=ConfigDict(strict=True)
                ).validate_python(self.default)
            except ValidationError as exc:
                raise ValueError(
                    f"field {self.name!r}: default {self.default!r} is not a {self.type}"
                ) from exc
        return self

    @property
    def has_default(self) -> bool:
        return self.default is not MISSING

    def annotation(self) -> Any:
        """The Python type this field's ``type`` stands for."""
        return parse_type(self.type)


def _unique_names(fields: list[FieldSpec]) -> list[FieldSpec]:
    seen: set[str] = set()
    for field in fields:
        if field.name in seen:
            raise ValueError(f"field {field.name!r} is defined more than once")
        seen.add(field.name)
    return fields


type FieldSpecs = Annotated[list[FieldSpec], AfterValidator(_unique_names)]
"""An ordered list of field specs with unique names."""

FIELD_SPECS: TypeAdapter[list[FieldSpec]] = TypeAdapter(FieldSpecs)


class OutputModelSpec(BaseModel):
    """An agent's structured output: a named model over a list of fields.

    The name becomes the class name and the description its docstring; both are what
    the LLM sees as the structured-output tool's name and description.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The model name, a Python identifier.")
    description: str | None = Field(
        default=None, description="What the output reports, shown to the LLM."
    )
    fields: FieldSpecs = Field(description="The output fields, in order.")

    @field_validator("name")
    @classmethod
    def _valid_name(cls, name: str) -> str:
        return _check_name(name)


def dump_fields(fields: Sequence[FieldSpec]) -> list[dict[str, Any]]:
    """Return field specs in their normalized JSON form."""
    return [field.model_dump(mode="json") for field in fields]


def build_input_model(fields: Sequence[FieldSpec]) -> type[BaseModel]:
    """Return the model that validates an agent's inputs against its arguments.

    Validation is strict: a bool is not an int and a string is not a number, but an int
    is a float. Unknown inputs are rejected.
    """
    return _input_model(json.dumps(dump_fields(fields)))


def build_output_model(spec: OutputModelSpec) -> type[TemplateModel]:
    """Return the class an agent with this output model reports."""
    return _output_model(spec.model_dump_json())


@cache
def _input_model(key: str) -> type[BaseModel]:
    return create_model(
        "AgentInputs",
        __config__=ConfigDict(strict=True, extra="forbid"),
        **_definitions(FIELD_SPECS.validate_json(key)),
    )


@cache
def _output_model(key: str) -> type[TemplateModel]:
    spec = OutputModelSpec.model_validate_json(key)
    return create_model(
        spec.name,
        __base__=TemplateModel,
        __doc__=spec.description,
        **_definitions(spec.fields),
    )


def _definitions(fields: Sequence[FieldSpec]) -> dict[str, Any]:
    return {
        field.name: (
            field.annotation(),
            Field(
                default=field.default if field.has_default else ...,
                description=field.description,
            ),
        )
        for field in fields
    }
