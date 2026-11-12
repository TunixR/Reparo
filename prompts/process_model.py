"""Shared prompt text on how the recovery payload links to the process model."""

MODEL_EVENT_GUIDANCE = """How the UI log maps onto `model`:
- A model event is a task defined in `model`; its model event id is the task's `id`, the same on every run and loop pass.
- `event_id` (in `ui_log` and `errored_act`) only identifies an occurrence in this run, counting up. It is never a task of `model`.
- `model_event_id` (nullable, in `ui_log` and `errored_act`) is the id of the task in `model` that the entry performed. Use it to find the model event an entry performed.
- Only when `model_event_id` is missing, match `event_name` against the task names in `model`, ignoring any trailing ` #<n>` suffixes. If more than one task matches, say that the match is ambiguous.
- `event_name` ends with one ` #<pass>` per loop that encloses the event, outermost first and 0-based (e.g. `Click Approve #1#2` is the second pass of the outer loop and the third of the inner one). Read the current loop passes off the suffix of `errored_act.event_name`; no suffix means the event is not inside a loop."""
