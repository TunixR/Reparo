# Technical debt

Shortcuts we know about and chose to leave for now. Each entry says where the debt is, why it is
debt and what paying it off means. When you fix one, delete its entry in the same PR; when you
knowingly leave one behind, add it here.

## 1. GUI agents always report `GuiRunReport`

- **Where:** `Agent.output_class` in `database/agents/models.py`; `GuiRunReport` in
  `database/agents/tooled_gui_agent.py`.
- **Why it's debt:** every other agent's output model is data, but a GUI agent reports a class
  built in code, whatever its `output_model` says. Graph conditions and bindings over a GUI node
  can only use `GuiRunReport`'s fields, and changing them needs a deploy.
- **What fixing it means:** let the GUI loop fill an output model defined as data (for example by
  asking the grounding model for a structured report at the end of its run), and fall back to
  `GuiRunReport` only when none is defined.

## 2. `tooled_guiagent` has a fixed signature

- **Where:** `TooledGuiAgent.tooled_guiagent` in `database/agents/tooled_gui_agent.py`.
- **Why it's debt:** the tool takes `task`, `variables`, `ui_log`, `errored_act` and `model` as
  Python parameters. A GUI agent's `arguments` only document those inputs: adding, renaming or
  retyping one changes nothing at runtime, or breaks the call.
- **What fixing it means:** build the GUI tool's input from the agent's `arguments`, as other
  agents do, and read the values the loop needs from the validated inputs.

## 3. No prompt templating

- **Where:** the instruction built in `Agent._execute` (`database/agents/models.py`).
- **Why it's debt:** an agent's inputs reach the LLM as `name: value` lines after the system
  prompt. A prompt can't place an input where it reads best, format it, or leave one out.
- **What fixing it means:** a template on the agent that names its arguments, validated against
  them when the agent is saved, with today's `name: value` lines as the default.

## 4. Seed argument types are too loose, and `errored_act` can't be null

- **Where:** the `arguments` of the seeded agents in `database/populators/agents.json`.
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

## 10. The `tooled_guiagent` tool returns the inner agent's trimmed history

- **Where:** the end of `TooledGuiAgent.tooled_guiagent` in `database/agents/tooled_gui_agent.py`,
  which returns every assistant message left in `run.agent.messages`.
- **Why it's debt:** the conversation manager edits that list in place. `GUI_WINDOW_SIZE` fits a
  run of `MAX_ACTIONS` actions, but `alter_variable` and `note_progress` calls don't count against
  the budget, so a longer run loses its first turns at the end-of-run trim. Proactive compression
  also drops old turns mid-run when the context fills up. The parent agent then gets a history
  that starts partway through, and the full history floods its context anyway.
- **What fixing it means:** return a condensed result built in code, such as the `GuiRunReport`
  the graph node already gets, so the window only has to manage the model's context.
