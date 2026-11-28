# Technical debt

Shortcuts we know about and chose to leave for now. Each entry says where the debt is, why it is
debt and what paying it off means. When you fix one, delete its entry in the same PR; when you
knowingly leave one behind, add it here.

## 1. GUI agents always report `GuiRunReport`

- **Where:** `GuiAgent.output_class` in `database/agents/behaviours/gui/base.py`; `GuiRunReport`
  in `database/agents/behaviours/gui/shared.py`.
- **Why it's debt:** every other agent's output model is data, but a GUI agent reports a class
  built in code, whatever its `output_model` says. Graph conditions and bindings over a GUI node
  can only use `GuiRunReport`'s fields, and changing them needs a deploy.
- **What fixing it means:** let the GUI loop fill an output model defined as data (for example by
  asking the grounding model for a structured report at the end of its run), and fall back to
  `GuiRunReport` only when none is defined.

## 2. GUI agents take fixed inputs

- **Where:** `GUI_ARGUMENTS` in `database/agents/behaviours/gui/shared.py`, which `GuiAgent.validate`
  writes over a GUI agent's `arguments`.
- **Why it's debt:** every GUI agent takes `task`, `variables`, `ui_log`, `errored_act` and `model`,
  because the loop and its instruction read those five by name. Arguments given on create, update
  or seed are replaced (and listed in `ignored_fields`), so a GUI agent can't take another input,
  and a graph must bind all five to a GUI node.
- **What fixing it means:** let the loop need only what it uses (`task`, and `variables` for
  `alter_variable`), and build the rest of the instruction from the agent's own arguments, with
  prompt templating (#3).

## 3. No prompt templating

- **Where:** the instruction built in `DefaultAgent.run` (`database/agents/behaviours/default.py`).
- **Why it's debt:** an agent's inputs reach the LLM as `name: value` lines after the system
  prompt. A prompt can't place an input where it reads best, format it, or leave one out.
- **What fixing it means:** a template on the agent that names its arguments, validated against
  them when the agent is saved, with today's `name: value` lines as the default.

## 4. Seed argument types are too loose, and `errored_act` can't be null

- **Where:** the `arguments` of the seeded agents in `database/populators/agents.json`. GUI agents
  are not affected: they take `GUI_ARGUMENTS` (#2), where `errored_act` is `dict | None`.
- **Why it's debt:** the seeds kept the old, bare `dict` and `list` types. They accept any shape,
  so a malformed input is only caught when the LLM misreads it. `errored_act` is null when the
  robot didn't name the failed activity (`RecoveryInputs`), but its seeded type doesn't allow
  None, so such a recovery fails input validation.
- **What fixing it means:** give the seeds precise types (`dict | None` for `errored_act`, item
  types for the lists), with a migration step or manual-steps entry for existing databases, since
  seeding never overwrites an agent.

## 5. Field types can't be nested objects

- **Where:** the type whitelist in `database/agents/specs.py`.
- **Why it's debt:** a field is a scalar, a list, a `dict[str, T]` or a union of those. An output
  that is naturally an object (a finding with a reason and tags) becomes a loose `dict`, which the
  LLM gets no schema for and graph conditions can't reach into.
- **What fixing it means:** let a field's type name a nested model defined in the same output
  model (or a shared one), with the same validation, normalization and caching.

## 6. The recovery input names are written in three places

- **Where:** `_recovery_inputs` in `routers/recovery.py`, `RecoveryInputs` in
  `templates/recovery_payload.py`, and the gateway's `arguments` in
  `database/populators/agents.json`.
- **Why it's debt:** the names and types of the recovery inputs must agree in all three. Renaming
  one in only one place breaks recoveries at runtime, not at startup.
- **What fixing it means:** one source, `RecoveryInputs`, from which the router builds the inputs
  and against which the seed (or a startup check) validates the gateway's arguments.

## 7. Graph binding checks argument names but not their types

- **Where:** `_input_map_violations` and `_path_error` in `database/agent_graphs/service.py`.
- **Why it's debt:** a node's input map must name its agent's arguments and point at paths that
  exist, but a path whose value has the wrong type (a `str` output bound to an `int` argument)
  passes validation and fails only when the node runs.
- **What fixing it means:** compare the type at the path with the argument's type when a graph is
  validated, reusing `field_type` (`database/agent_graphs/field_types.py`), which already resolves them for
  conditions.

## 8. Traces don't store an agent's structured output

- **Where:** `AgentTrace` and `AgentGraphNodeTrace` in `database/logging/models.py`.
- **Why it's debt:** traces keep an agent's inputs and messages, but not the output model instance
  it reported. Reading what a node decided (and why a graph took an edge) means digging the
  structured-output tool call out of the messages.
- **What fixing it means:** store the reported output as JSON on the agent trace (or node trace),
  and show it in the trace log.

## 9. The eval connection refuses an invalid payload with its own code

- **Where:** `EVAL_VALIDATION_ERROR` in `routers/evals.py`, passed to the shared
  `_validate_payload_or_raise` (`routers/recovery.py`).
- **Why it's debt:** the eval connection answers an invalid payload, or one without a recovery
  task, with `VALIDATION_ERROR`; the recovery connection uses `INVALID_RECOVERY_PAYLOAD` for the
  same payload. Robots have to match two codes for one ending. The Playwright Bridge matches on
  `VALIDATION_ERROR`, so the rename is a breaking contract change (ADR 0001).
- **What fixing it means:** drop `EVAL_VALIDATION_ERROR`, so the eval connection sends
  `INVALID_RECOVERY_PAYLOAD`, as part of the next `contract_version` bump, through the Contract changes checklist in ReparoHome's
  `docs/agents/issue-tracker.md`, and update the eval line in `ARCHITECTURE.md` "Robot protocol".

## 10. Holo models are told apart by their name

- **Where:** `GuiAgent._observation` in `database/agents/behaviours/gui/base.py`.
- **Why it's debt:** a Holo model gets each screenshot after an action between `<observation>`
  tags, and the code finds one by looking for "Holo" in the router's `model_name`. Another name
  for the same model, or another model that needs the tags, gets the wrong framing silently.
- **What fixing it means:** a property of the router (like its `vision_backbone`), set when it is
  configured, that says how the model wants observations framed.

## 11. Code the static analysis tools already complain about

- **Where:** the debt lists in `pyproject.toml`: the import contracts' `ignore_imports` (the upward
  imports from `agent_tools` and `observability` into `database`, and the imports that close the
  cycles inside `database`), and ruff's per-file ignore for migration 0008's `print` calls.
- **Why it's debt:** the tools were turned on without changing the code, and these findings need
  more than a local fix. The upward imports and the cycles are what make `database` and the
  hook/telemetry packages depend on each other. The migration reports what it changed with `print`;
  logging instead must still reach whoever runs the upgrade (`alembic` or `populate.py`).
- **What fixing it means:** move what `agent_tools.hooks` and the telemetry modules need from
  `database` behind an interface they own (or move the code down a layer), break the cycles in
  `database.agent_graphs.runner`, report migration 0008's changes through logging, and delete each
  line as its finding goes. Then this entry goes too.

## 12. An agent trace's messages are one JSONB list, rewritten on every message

- **Where:** `AgentLoggingHook.update_trace` in `agent_tools/hooks.py`, and `AgentTrace.messages`
  in `database/logging/models.py`.
- **Why it's debt:** every message an agent adds writes the whole list again, so a run of n
  messages writes O(n²) data. Appending in SQL (`messages || :new`) would only cut the JSON
  encoding and the bytes sent, not the write: Postgres rewrites a whole JSONB value on any change,
  and an earlier message still changes when its screenshot's S3 uuid is filled in, which needs a
  full rewrite. The write is one UPDATE without a read.
- **What fixing it means:** store each message as a row of its own (agent trace, position,
  content), inserted once and updated only to fill in a screenshot, with a migration of the
  existing lists and of the readers (`routers/logging.py`, the markdown export).

## 13. Trace writes block the event loop

- **Where:** the hooks in `agent_tools/hooks.py` and `AgentGraphRunTrace` in
  `database/logging/agent_graph_tracing.py`, called from Strands callbacks and the graph runner.
- **Why it's debt:** each write is a synchronous round trip to Postgres on the event loop thread,
  so every other recovery and request served by the process waits for it. It is one statement per
  write, but a slow database stalls the whole process.
- **What fixing it means:** run the writes off the loop (a thread or an async engine), keeping
  their order per trace, since a node trace links to an agent trace inserted just before it.

## 14. Agent trace lists carry each trace's whole conversation

- **Where:** `list_agent_traces` in `routers/logging.py`, which answers `AgentTrace` rows as they
  are stored.
- **Why it's debt:** each listed trace brings its `messages` (the whole LLM conversation, often
  with image references), although a list is for picking a trace to open. A page of up to 1000
  traces can be megabytes of JSON the caller doesn't read.
- **What fixing it means:** a list read model without `messages`, with the full trace left to
  `GET /logging/agent_traces/{trace_id}`. It changes the list's
  response shape, so clients that read `messages` from the list must move to the detail call.
