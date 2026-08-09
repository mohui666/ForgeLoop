# Edit Intent Handoff v1

Edit Intent Handoff is a small protocol layer in front of Hybrid Controller
v1.2 implementation. It addresses a specific failure: V4-Flash can reach the
`implement` state without having formed an executable modification plan, then
spend the remaining budget asking for more exploration.

The bundled policy is `deepseek-v4-flash-edit-intent-v1`. It keeps:

- V4-Flash as the only coding and edit-planning model;
- local `qwen2.5:1.5b-instruct` as a state classifier only;
- deterministic Controller v1 for hard limits and terminal decisions;
- Hybrid Controller v1.2 tool gating after intent acceptance;
- the existing Agent tool protocol and unchanged DeepSWE collector/verifier.

No SFT, RL, planner, critic, second coding model or benchmark change is part of
this implementation.

## Handoff protocol

When the classifier first selects `explore -> implement`, ForgeLoop keeps the
effective state in `explore` and asks V4-Flash to call the Controller-owned
`submit_edit_intent` tool. Its strict schema is:

```text
target_files: 1-4 unique concrete paths
diagnosis: non-empty string
intended_change: non-empty string
validation_command: non-empty string
```

Every target is resolved through the selected Workspace/Runtime and must be an
existing non-sensitive file. The three text fields have explicit compactness
limits. The validated intent is recorded as `edit_intent_accepted`, including
`schema_valid=true` and `target_files_exist=true`.

The tool observation then returns the same intent as compact working context:

```text
target_files: ...
diagnosis: ...
intended_change: ...
validation_command: ...
```

Only then does the effective state become `implement`, where the unchanged
v1.2 gates apply.

This is a handoff rather than a stronger exploration gate. While intent is
pending, V4-Flash retains a bounded read-only context window of at most three
`read_file`, `search_files`, `list_files` or non-history `git_inspect` actions.
Mutating tools are unavailable until an intent is accepted. An invalid or
missing intent exposes exactly one focused `read_file`/`search_files` replan.
A second invalid/missing handoff ends the run with
`controller_invalid_edit_intent` instead of consuming the full task budget.

Multiple tool calls already emitted in the model response that caused the
classifier transition are completed before the handoff activates. This matters
because those calls were selected from the previous explore schema; counting a
sibling call as an invalid intent was an orchestration bug found during the
first live regression attempt.

## Trajectory events

- `edit_intent_requested`
- `edit_intent_handoff_activated`
- `edit_intent_context_action`
- `edit_intent_accepted`
- `edit_intent_rejected`
- `edit_intent_focused_replan`

The run summary preserves request/accept/reject counts, context actions,
focused replan usage and the accepted intent itself.

## Run

```powershell
$env:DEEPSEEK_API_KEY = "..."
ollama pull qwen2.5:1.5b-instruct
uv run forgeloop eval --stage c --task query-encoding --live `
  --repeats 1 `
  --policy-manifest deepseek-v4-flash-edit-intent-v1

uv run forgeloop deepswe run `
  --task oxvg-structural-selector-preservation `
  --policy deepseek-v4-flash-edit-intent-v1
```

## Context audit

The previous v1.2 oxvg trajectory showed that the first V4 request received:

- the complete DeepSWE task statement;
- the `/app` POSIX/Docker runtime description;
- the Pier commit/patch-collection instruction;
- all ordinary ForgeLoop tool definitions.

Before the classifier selected `implement`, however, V4 had observed only Git
status/branch/history and the repository-root listing. It had not read source
code or identified an implementation file. The task and runtime context were
present; the defect was presenting an implement transition as actionable before
V4 had committed to targets and a diagnosis.

The handoff fixes that presentation boundary. No repository text is sent to the
1.5B classifier, and the Agent system/task prompt is otherwise unchanged.

## Real validation on 2026-08-09

### Frozen `query-encoding`

The final LocalRuntime run passed and ended `completed/model_finish_tool`:

- 7 model calls, 9 tool calls;
- 20,859 input and 2,145 output tokens, 23,004 total;
- 31.289 seconds, provider cost $0.00164177;
- intent accepted with target `urls.py`, a concrete URL-encoding diagnosis,
  an `urlencode(..., doseq=True)` change and
  `python -m unittest tests.test_urls -v` validation;
- actual progression: context reads -> intent -> patch -> passing test -> diff
  -> finish.

This satisfies the frozen regression and proves that accepted intent is
recorded and returned as usable working context.

Two earlier development attempts failed at 4,699 and 4,630 tokens. They exposed
the same-response activation bug and an over-restrictive intent-only schema.
Both were fixed with regression tests; they are not counted as successful
evaluation results.

### DeepSWE `oxvg-structural-selector-preservation`

One final run used pinned DeepSWE revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0, its official Docker
environment and unchanged verifier/patch collector. It failed honestly:

- terminal: `failed/controller_invalid_edit_intent`;
- 5 model calls, 8 tool calls;
- 34,159 input and 996 output tokens, 35,155 total;
- 154.812 seconds, provider cost $0.0031118;
- 3 bounded context actions and 1 focused replan;
- 2 rejected/missing handoffs; no valid intent was generated;
- no `apply_patch`, no test action and no patch collected;
- verifier: P2P 62/62, F2P 0/6, partial 0.9117647058823529, reward 0.

The prior final v1.2 attempt consumed 200,049 tokens and ended at the budget
guard without a patch. Edit Intent Handoff reduced this failed attempt by about
82% and produced a precise terminal reason, but it did not improve solve
ability on oxvg. The failure remains `model_failure`: V4-Flash continued asking
for context and never submitted the required structured intent. No generalized
improvement claim is made.
