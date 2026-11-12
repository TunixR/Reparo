"""Read the process model a robot sends with its recovery payload.

These are plain utilities, not agent tools: the continuation tool uses them to
check the model event id and loop passes an agent names.

`model` is BPMN 2.0 XML or another textual format (e.g. an indented outline).
For BPMN, every task-like element is a model event whose `id` is the model
event id. For any other format, only a substring check is possible.

Loop depth (#231) is read from two kinds of loops:
- a looping `subProcess` (`standardLoopCharacteristics`, or a sequential
  `multiInstanceLoopCharacteristics`) encloses everything inside it;
- a gateway cycle: a DFS over the sequence flows from the start events finds
  the back-edges `u -> h`, and the natural loop of `h` is `h` plus every node
  that reaches `u` without passing through `h`. Back-edges sharing a head are
  one loop. In a reducible graph, natural loops are nested or disjoint.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

# Only the Element type comes from xml.etree; parsing goes through defusedxml.
from xml.etree.ElementTree import Element  # nosec B405

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring

BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"

TASK_TAGS = frozenset(
    {
        "task",
        "userTask",
        "serviceTask",
        "manualTask",
        "scriptTask",
        "sendTask",
        "receiveTask",
        "businessRuleTask",
        "callActivity",
    }
)
SUBPROCESS_TAGS = frozenset({"subProcess", "transaction", "adHocSubProcess"})
# Children of a scope that are not flow nodes.
_NON_NODE_TAGS = frozenset(
    {
        "sequenceFlow",
        "laneSet",
        "documentation",
        "extensionElements",
        "ioSpecification",
        "property",
        "dataObject",
        "dataObjectReference",
        "dataStoreReference",
        "association",
        "textAnnotation",
        "group",
        "standardLoopCharacteristics",
        "multiInstanceLoopCharacteristics",
        "incoming",
        "outgoing",
    }
)
# How many valid tasks an error message lists before summarising the rest.
_MAX_LISTED_TASKS = 50


class ModelEventError(ValueError):
    """The id is not a model event of the process model."""


@dataclass(frozen=True)
class ModelEvent:
    """A task of the process model."""

    id: str
    name: str | None


@dataclass(frozen=True)
class Loop:
    """A loop enclosing a model event."""

    id: str
    name: str | None
    kind: Literal["subprocess", "gateway_cycle"]


LoopingKind = Literal["none", "loop", "unsupported"]


@dataclass
class Scope:
    """A process or subprocess: the flow nodes and sequence flows directly inside it."""

    id: str
    nodes: dict[str, str | None] = field(default_factory=dict)
    start_events: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ProcessModel:
    """A parsed BPMN process model.

    Attributes:
        tasks: Model events by id, in document order.
        scopes: Every process and subprocess by id.
        parent_scope: The scope directly containing each flow node.
        looping: How each activity loops: not at all, as a loop
            (standard or sequential multi-instance), or in an unsupported way
            (parallel multi-instance).
    """

    tasks: dict[str, ModelEvent] = field(default_factory=dict)
    scopes: dict[str, Scope] = field(default_factory=dict)
    parent_scope: dict[str, str] = field(default_factory=dict)
    looping: dict[str, LoopingKind] = field(default_factory=dict)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _bpmn_children(element: Element) -> Iterator[Element]:
    for child in element:
        if isinstance(child.tag, str) and child.tag.startswith(f"{{{BPMN_NS}}}"):
            yield child


def _looping_kind(element: Element) -> LoopingKind:
    for child in _bpmn_children(element):
        tag = _local(child.tag)
        if tag == "standardLoopCharacteristics":
            return "loop"
        if tag == "multiInstanceLoopCharacteristics":
            sequential = child.get("isSequential", "false").strip().lower() == "true"
            return "loop" if sequential else "unsupported"
    return "none"


def _read_scope(model: ProcessModel, element: Element) -> None:
    scope_id = element.get("id") or f"<{_local(element.tag)}>"
    scope = Scope(id=scope_id)
    model.scopes.setdefault(scope_id, scope)
    boundary_events: list[tuple[str, str]] = []

    for child in _bpmn_children(element):
        tag = _local(child.tag)
        if tag == "sequenceFlow":
            source, target = child.get("sourceRef"), child.get("targetRef")
            if source and target:
                scope.edges.append((source, target))
            continue
        if tag in _NON_NODE_TAGS:
            continue
        node_id = child.get("id")
        if not node_id:
            continue

        scope.nodes.setdefault(node_id, child.get("name"))
        model.parent_scope.setdefault(node_id, scope_id)
        if tag == "startEvent":
            scope.start_events.append(node_id)
        elif tag == "boundaryEvent" and (attached := child.get("attachedToRef")):
            boundary_events.append((attached, node_id))
        if tag in TASK_TAGS:
            model.tasks.setdefault(node_id, ModelEvent(node_id, child.get("name")))
        if tag in TASK_TAGS or tag in SUBPROCESS_TAGS:
            model.looping.setdefault(node_id, _looping_kind(child))
        if tag in SUBPROCESS_TAGS:
            _read_scope(model, child)

    # A boundary event leaves the activity it is attached to.
    scope.edges.extend(boundary_events)


def parse_model(model: str) -> ProcessModel | None:
    """Parse `model` as BPMN 2.0.

    Returns:
        The parsed model, or `None` when `model` is not BPMN 2.0 XML (e.g. a
        textual outline).
    """
    try:
        root = fromstring(model.strip())
    except (ParseError, DefusedXmlException, ValueError):
        return None
    if not isinstance(root.tag, str) or not root.tag.startswith(f"{{{BPMN_NS}}}"):
        return None

    parsed = ProcessModel()
    processes = (
        [root]
        if _local(root.tag) == "process"
        else list(root.iter(f"{{{BPMN_NS}}}process"))
    )
    for process in processes:
        _read_scope(parsed, process)
    return parsed


def _describe_tasks(tasks: Iterable[ModelEvent]) -> str:
    described = [f"{t.id} ({t.name})" if t.name else t.id for t in tasks]
    if not described:
        return "the model defines no tasks"
    extra = len(described) - _MAX_LISTED_TASKS
    listed = ", ".join(described[:_MAX_LISTED_TASKS])
    return f"valid tasks: {listed}" + (f", and {extra} more" if extra > 0 else "")


def check_model_event(model: str, model_event_id: str) -> None:
    """Check that `model_event_id` names a task of `model`.

    For BPMN, the id must be the id of a task-like element. For any other
    format, it must appear in `model` as a substring.

    Raises:
        ModelEventError: The id is not a task of the model. For BPMN, the
            message lists the valid task ids with their names.
    """
    if not model_event_id.strip():
        raise ModelEventError("The model event id is empty.")

    parsed = parse_model(model)
    if parsed is None:
        if model_event_id not in model:
            raise ModelEventError(
                f"{model_event_id!r} does not appear in the model. Use the id of a "
                "task exactly as the model writes it, not a UI log event_id."
            )
        return

    if model_event_id not in parsed.tasks:
        raise ModelEventError(
            f"{model_event_id!r} is not a task of the model; "
            f"{_describe_tasks(parsed.tasks.values())}."
        )


def describe_model_event(model: str, model_event_id: str) -> ModelEvent | None:
    """Return the BPMN task named `model_event_id`, or `None` if there is none."""
    parsed = parse_model(model)
    if parsed is None:
        return None
    return parsed.tasks.get(model_event_id)


def _natural_loops(scope: Scope) -> list[tuple[str, set[str]]] | None:
    """Find the gateway-cycle loops of one scope.

    Returns:
        `(header, body)` pairs, or `None` when the graph is irreducible.
    """
    successors: dict[str, list[str]] = defaultdict(list)
    predecessors: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set(scope.nodes)
    for source, target in scope.edges:
        successors[source].append(target)
        predecessors[target].append(source)
        nodes.update((source, target))

    roots = [n for n in scope.start_events if n in nodes]
    roots += sorted(n for n in nodes if not predecessors[n] and n not in roots)

    # Iterative DFS, recording back-edges (target still on the stack).
    on_stack: set[str] = set()
    visited: set[str] = set()
    back_edges: list[tuple[str, str]] = []
    for root in roots:
        if root in visited:
            continue
        visited.add(root)
        on_stack.add(root)
        stack: list[tuple[str, Iterator[str]]] = [(root, iter(successors[root]))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                on_stack.discard(node)
                _ = stack.pop()
            elif child in on_stack:
                back_edges.append((node, child))
            elif child not in visited:
                visited.add(child)
                on_stack.add(child)
                stack.append((child, iter(successors[child])))

    if visited != nodes:
        # A cycle with no way in from a start: its depth can't be read.
        return None

    dominators = _dominators(visited, roots, predecessors)
    loops: dict[str, set[str]] = {}
    for tail, header in back_edges:
        if header not in dominators[tail]:
            return None  # Irreducible: a cycle entered at more than one node.
        body = loops.setdefault(header, {header})
        pending = [tail]
        while pending:
            node = pending.pop()
            if node in body:
                continue
            body.add(node)
            pending.extend(predecessors[node])
    return list(loops.items())


def _dominators(
    nodes: set[str], roots: list[str], predecessors: dict[str, list[str]]
) -> dict[str, set[str]]:
    """Compute dominator sets, with a virtual entry above every root."""
    entry = "\0entry"
    preds = {n: list(predecessors[n]) for n in nodes}
    for root in roots:
        preds[root].append(entry)
    dom: dict[str, set[str]] = {n: set(nodes) | {entry} for n in nodes}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            new = set.intersection(*(dom[p] for p in preds[node])) | {node}
            if new != dom[node]:
                dom[node] = new
                changed = True
    return dom


def enclosing_loops(model: str, model_event_id: str) -> list[Loop] | None:
    """Return the loops enclosing a model event, outermost first.

    Returns:
        The loops, `[]` when the event is in no loop, or `None` when the depth
        is unknown: `model` is not BPMN, the id is not a task, a scope is
        irreducible, or the event is inside a parallel multi-instance activity
        (out of scope, #231).
    """
    parsed = parse_model(model)
    if parsed is None or model_event_id not in parsed.tasks:
        return None

    # The chain of elements from the outermost scope down to the task:
    # each element is a flow node of the scope it is paired with.
    chain: list[tuple[str, str]] = []
    element = model_event_id
    while (scope_id := parsed.parent_scope.get(element)) is not None:
        chain.append((scope_id, element))
        if scope_id not in parsed.parent_scope:
            break
        element = scope_id
    chain.reverse()

    loops: list[Loop] = []
    for scope_id, element in chain:
        scope = parsed.scopes[scope_id]
        natural = _natural_loops(scope)
        if natural is None:
            return None
        cycles = sorted(
            ((header, body) for header, body in natural if element in body),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
        loops.extend(
            Loop(header, scope.nodes.get(header), "gateway_cycle")
            for header, _ in cycles
        )

        looping = parsed.looping.get(element, "none")
        if looping == "unsupported":
            return None
        if looping == "loop":
            if element == model_event_id:
                # A looping task: whether it encloses itself isn't defined (#231).
                return None
            loops.append(Loop(element, scope.nodes.get(element), "subprocess"))
    return loops
