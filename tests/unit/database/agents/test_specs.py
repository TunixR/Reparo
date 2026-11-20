"""Agent arguments and output models defined as data: `FieldSpec` and `OutputModelSpec`."""

from typing import Any

import pytest
from pydantic import ValidationError
from strands.tools.structured_output.structured_output_utils import (
    convert_pydantic_to_tool_spec,
)

from database.agents.specs import (
    FieldSpec,
    OutputModelSpec,
    build_input_model,
    build_output_model,
)
from templates.common import TemplateModel


def _field(type_: str, **extra: Any) -> FieldSpec:
    return FieldSpec.model_validate(
        {"name": "value", "type": type_, "description": "A value."} | extra
    )


def _accepts(type_: str, value: Any) -> bool:
    model = build_input_model([_field(type_)])
    try:
        _ = model.model_validate({"value": value})
    except ValidationError:
        return False
    return True


@pytest.mark.parametrize(
    ("type_", "good", "bad"),
    [
        ("str", "text", 1),
        ("int", 3, "3"),
        ("float", 1.5, "1.5"),
        ("bool", True, 1),
        ("None", None, 0),
        ("list[int]", [1, 2], ["1"]),
        ("list", [1, "a"], {"a": 1}),
        ("dict[str, int]", {"a": 1}, {"a": "1"}),
        ("dict", {"a": [1]}, [1]),
        ("str | None", None, 1),
        ("int | str", "a", 1.5),
        ("Literal['navigation', 'other']", "other", "typo"),
        ("Literal[1, -2, True, None]", -2, 3),
        ("list[dict[str, Any]]", [{"k": "v"}], [1]),
    ],
)
def test_each_whitelisted_type_validates_its_values(
    type_: str, good: Any, bad: Any
) -> None:
    assert _accepts(type_, good)
    assert not _accepts(type_, bad)


def test_an_int_passes_a_float_field() -> None:
    assert _accepts("float", 3)


def test_any_takes_every_value() -> None:
    assert _accepts("Any", {"any": ["thing"]})
    assert _accepts("Any", None)


@pytest.mark.parametrize(
    ("type_", "hint"),
    [
        ("Optional[str]", "str | None"),
        ("Union[int, str]", "int | str"),
        ("tuple[int]", "tuple"),
        ("set[str]", "set"),
        ("dict[int, str]", "keys"),
        ("list[int, str]", "list"),
        ("Literal[str]", "Literal"),
        ("Literal[()]", "Literal"),
        ("bytes", "bytes"),
        ("GuiRunReport", "GuiRunReport"),
        ("__import__('os')", "__import__"),
        ("str.__class__", "str.__class__"),
        ("int + str", "+"),
        ("lambda: 1", "lambda"),
        ("not a type", "not a type"),
        ("", "type"),
    ],
)
def test_a_type_outside_the_whitelist_is_rejected(type_: str, hint: str) -> None:
    with pytest.raises(ValidationError) as info:
        _ = _field(type_)

    message = str(info.value)
    assert "value" in message
    assert hint in message


def test_a_field_without_default_is_required_even_when_nullable() -> None:
    model = build_input_model([_field("str | None")])

    with pytest.raises(ValidationError, match="value"):
        _ = model.model_validate({})
    assert model.model_validate({"value": None}).model_dump() == {"value": None}


def test_a_defaulted_field_is_optional() -> None:
    model = build_input_model([_field("list[int]", default=[1]), _field_named("other")])

    assert model.model_validate({"other": "x"}).model_dump() == {
        "value": [1],
        "other": "x",
    }


def _field_named(name: str, type_: str = "str") -> FieldSpec:
    return FieldSpec(name=name, type=type_, description=f"The {name}.")


@pytest.mark.parametrize(
    ("type_", "default"),
    [
        ("int", "3"),
        ("str", None),
        ("bool", 1),
        ("Literal['a']", "b"),
        ("list[int]", ["1"]),
    ],
)
def test_a_default_must_be_a_value_of_its_type(type_: str, default: Any) -> None:
    with pytest.raises(ValidationError, match="field 'value': default"):
        _ = _field(type_, default=default)


def test_a_null_default_makes_a_nullable_field_optional() -> None:
    spec = _field("str | None", default=None)

    assert spec.has_default
    assert build_input_model([spec]).model_validate({}).model_dump() == {"value": None}


@pytest.mark.parametrize(
    "name", ["1st", "has space", "class", "_private", "model_x", ""]
)
def test_a_field_name_must_be_a_public_identifier(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        _ = _field("str", name=name)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="json_type"):
        _ = _field("str", json_type="string")
    with pytest.raises(ValidationError, match="extra"):
        _ = OutputModelSpec.model_validate({"name": "Report", "fields": [], "extra": 1})


def test_field_names_are_unique_within_an_output_model() -> None:
    with pytest.raises(ValidationError, match="'value' is defined more than once"):
        _ = OutputModelSpec(name="Report", fields=[_field("str"), _field("int")])


def test_an_output_model_name_must_be_an_identifier() -> None:
    with pytest.raises(
        ValidationError, match="'Gateway Decision' is not a Python identifier"
    ):
        _ = OutputModelSpec(name="Gateway Decision", fields=[])


def test_a_definition_is_stored_in_a_normalized_form() -> None:
    spec = FieldSpec.model_validate(
        {"name": "kind", "type": "Literal[\"a\",'b']|None", "description": "Kind."}
    )

    assert spec.model_dump(mode="json") == {
        "name": "kind",
        "type": "Literal['a', 'b'] | None",
        "description": "Kind.",
    }
    assert _field("int", default=1).model_dump()["default"] == 1


REPORT = OutputModelSpec(
    name="TriageReport",
    description="How the failure was triaged.",
    fields=[
        _field_named("kind", "Literal['ui', 'data']"),
        _field_named("note", "str | None"),
        FieldSpec(name="tries", type="int", description="Tries.", default=0),
    ],
)


def test_an_output_model_builds_a_named_documented_template_model() -> None:
    report = build_output_model(REPORT)

    assert issubclass(report, TemplateModel)
    assert report.__name__ == "TriageReport"
    assert report.__doc__ == "How the failure was triaged."
    assert report.model_validate({"kind": "ui", "note": None}).model_dump() == {
        "kind": "ui",
        "note": None,
        "tries": 0,
    }
    with pytest.raises(ValidationError):
        _ = report.model_validate({"kind": "ui", "note": None, "unknown": 1})


def test_the_llm_sees_the_output_model_name_and_description_as_the_tool() -> None:
    tool_spec = convert_pydantic_to_tool_spec(build_output_model(REPORT))

    assert tool_spec["name"] == "TriageReport"
    assert tool_spec["description"] == "How the failure was triaged."
    schema = tool_spec["inputSchema"]["json"]
    assert schema["properties"]["kind"]["enum"] == ["ui", "data"]


def test_built_classes_are_cached_by_their_normalized_definition() -> None:
    same = OutputModelSpec.model_validate_json(REPORT.model_dump_json())
    spaced = [_field("str|None")]

    assert build_output_model(same) is build_output_model(REPORT)
    assert build_input_model(spaced) is build_input_model([_field("str | None")])
