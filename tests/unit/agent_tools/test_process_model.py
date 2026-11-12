import pytest

from agent_tools.process_model import (
    Loop,
    ModelEventError,
    check_model_event,
    enclosing_loops,
    parse_model,
)
from tests.unit.agent_tools._shared.process_models import (
    COMPACT_MODEL,
    IRREDUCIBLE_MODEL,
    LOOPING_SUBPROCESS_MODEL,
    NESTED_LOOPS_MODEL,
    PARALLEL_MULTI_INSTANCE_MODEL,
    SPENDING_APPROVAL_MODEL,
)

# ---------------------------------------------------------------------------
# parse_model
# ---------------------------------------------------------------------------


def test_parse_model_reads_the_bpmn_example():
    parsed = parse_model(SPENDING_APPROVAL_MODEL)

    assert parsed is not None
    assert list(parsed.tasks) == [
        "open_requests",
        "open_request",
        "click_approve",
        "click_reject",
        "back_to_list",
    ]
    assert parsed.tasks["click_approve"].name == "Click Approve"
    process = parsed.scopes["spending_approval"]
    assert process.start_events == ["start"]
    assert ("each_request_next", "each_request") in process.edges
    assert "each_request" in process.nodes
    assert parsed.parent_scope["click_approve"] == "spending_approval"


def test_parse_model_collects_every_task_like_element():
    model = """<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
      <process id="p">
        <userTask id="a"/><serviceTask id="b"/><manualTask id="c"/>
        <scriptTask id="d"/><sendTask id="e"/><receiveTask id="f"/>
        <businessRuleTask id="g"/><callActivity id="h"/>
        <exclusiveGateway id="gw"/><startEvent id="s"/>
      </process>
    </definitions>"""

    parsed = parse_model(model)

    assert parsed is not None
    assert list(parsed.tasks) == ["a", "b", "c", "d", "e", "f", "g", "h"]


def test_parse_model_reads_tasks_inside_subprocesses():
    parsed = parse_model(LOOPING_SUBPROCESS_MODEL)

    assert parsed is not None
    assert "click_approve" in parsed.tasks
    assert parsed.parent_scope["click_approve"] == "each_request"
    assert parsed.parent_scope["each_request"] == "spending_approval"


@pytest.mark.parametrize(
    "model",
    [
        COMPACT_MODEL,
        "",
        "not xml at all <",
        "<definitions><process id='p'><task id='a'/></process></definitions>",
        '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "boom">]><x>&a;</x>',
    ],
    ids=["compact", "empty", "broken-xml", "xml-without-bpmn-namespace", "entities"],
)
def test_parse_model_returns_none_for_non_bpmn(model: str):
    assert parse_model(model) is None


# ---------------------------------------------------------------------------
# check_model_event
# ---------------------------------------------------------------------------


def test_check_model_event_accepts_a_bpmn_task():
    check_model_event(SPENDING_APPROVAL_MODEL, "click_approve")


@pytest.mark.parametrize(
    "model_event_id", ["click_aprove", "8", "each_request", "start"]
)
def test_check_model_event_rejects_what_is_not_a_bpmn_task(model_event_id: str):
    with pytest.raises(ModelEventError) as exc_info:
        check_model_event(SPENDING_APPROVAL_MODEL, model_event_id)

    message = str(exc_info.value)
    assert f"{model_event_id!r} is not a task of the model" in message
    assert "click_approve (Click Approve)" in message
    assert "back_to_list (Return to the list)" in message


def test_check_model_event_falls_back_to_a_substring_on_a_compact_model():
    check_model_event(COMPACT_MODEL, "click_approve")

    with pytest.raises(ModelEventError, match="does not appear in the model"):
        check_model_event(COMPACT_MODEL, "click_aprove")


def test_check_model_event_rejects_an_empty_id():
    with pytest.raises(ModelEventError, match="empty"):
        check_model_event(COMPACT_MODEL, "  ")


# ---------------------------------------------------------------------------
# enclosing_loops
# ---------------------------------------------------------------------------


def test_enclosing_loops_on_the_bpmn_example():
    assert enclosing_loops(SPENDING_APPROVAL_MODEL, "click_approve") == [
        Loop("each_request", "For each request_id in pending", "gateway_cycle")
    ]
    assert enclosing_loops(SPENDING_APPROVAL_MODEL, "open_requests") == []


def test_enclosing_loops_on_two_nested_gateway_cycles():
    outer = Loop("each_approver", "For each approver", "gateway_cycle")
    inner = Loop("each_request", "For each request of the approver", "gateway_cycle")

    assert enclosing_loops(NESTED_LOOPS_MODEL, "click_approve") == [outer, inner]
    assert enclosing_loops(NESTED_LOOPS_MODEL, "open_approver") == [outer]
    assert enclosing_loops(NESTED_LOOPS_MODEL, "log_in") == []


def test_enclosing_loops_on_a_looping_subprocess():
    assert enclosing_loops(LOOPING_SUBPROCESS_MODEL, "click_approve") == [
        Loop("each_request", "For each request", "subprocess")
    ]
    assert enclosing_loops(LOOPING_SUBPROCESS_MODEL, "open_requests") == []


@pytest.mark.parametrize(
    "model, model_event_id",
    [
        (COMPACT_MODEL, "click_approve"),
        (SPENDING_APPROVAL_MODEL, "click_aprove"),
        (IRREDUCIBLE_MODEL, "a"),
        (PARALLEL_MULTI_INSTANCE_MODEL, "click_approve"),
    ],
    ids=["non-bpmn", "unknown-task", "irreducible", "parallel-multi-instance"],
)
def test_enclosing_loops_is_unknown(model: str, model_event_id: str):
    assert enclosing_loops(model, model_event_id) is None
