# Execution Closure v2

Execution Closure v2 replaces the accumulated post-edit guards with one
deterministic delivery state machine. It closes the boundary between model work,
validation, finalization, Git delivery, and the DeepSWE collector without moving
code generation into the Controller.

The intended successful path is finite:

```text
source edit
  -> relevant explicit validation
  -> validation PASS for the exact current tree
  -> complete diff review
  -> finish (or one controller auto-finish decision)
  -> base-to-HEAD delivery commit
  -> non-empty collector model.patch
  -> official verifier
```

If review causes another edit, the validation and review evidence is invalidated
and the new tree must pass validation again.

## Audit findings

The 2026-08-11 audit covered the recent SQLite and Textual DeepSWE trajectories,
`AgentLoop`, Controller v1/v1.3, Pier runtime integration, and the official patch
collector contract.

The failures were not one missing prompt or readiness rule:

1. The collector creates `model.patch` from `git diff --binary <base> HEAD`.
   An uncommitted working tree can contain the correct solution and still produce
   an empty collector artifact.
2. Token enforcement happened while recording a returned model response. A
   response containing the needed validation or finish call could be discarded
   before its tool calls executed.
3. Validation was inferred from broad shell regexes. Import probes and commands
   that could edit the tree were able to look like successful validation.
4. The progress fingerprint was HEAD-relative. A legitimate commit could make
   apparent progress disappear even though the base-to-HEAD patch still existed.
5. Multiple post-edit reserves, action gates, scoped-read allowances, and finish
   rules competed after PASS. The model could be told to review or finish while a
   different guard blocked the required action, then eventually hit the budget.
6. Same-response edits were not coherent: after the first patch changed the tree,
   the next patch in the same model response could be rejected against a newly
   activated gate.
7. Controller telemetry used the last observed tool budget, so PASS-to-finish
   calls and tokens could omit the terminal model response.

The audited pre-v2 SQLite trajectory had a passing probe at 89,559 cumulative
tokens and then used roughly 83,000 more tokens before `budget_guard`. The
Textual trajectory failed once, passed its retest at 137,404 cumulative tokens,
then used roughly 64,700 more tokens before the same terminal. Neither delivered
a collector patch.

## Architecture

### Explicit evidence state

Controller v1.3 now tracks these execution phases:

```text
explore -> needs_validation -> validation_failed -> needs_review
        -> ready_to_finish -> terminal
```

The qwen2.5:1.5b classifier still runs and its decisions remain in trajectory
telemetry, but `classifier_advisory_only=true`: its output does not authorize or
block reads, edits, validation, review, or finish.

Before a source diff exists, normal tool schemas remain available. At 50% of the
token allocation the Controller emits one advisory convergence message. At 75%,
if no source diff exists, it terminates with
`controller_exploration_exhausted`. The limit is checked between model calls, so
one already-returned response can cross the threshold but is never discarded.

After a source diff exists:

- only the explicit `validate` tool, or a strict known test-runner command, can
  produce validation evidence;
- a validation command that changes the deliverable tree invalidates itself;
- failure moves to `validation_failed` and requires a fix plus retest;
- PASS is locked to the exact content fingerprint;
- a complete `git_diff`/Git diff review of that same tree is required;
- any later edit clears PASS and review evidence;
- `ready_to_finish` grants one model decision; if the tree is unchanged after
  that decision, the Controller deterministically auto-finishes;
- validation, repair, review, and post-PASS finalization each have finite call and
  token allocations.

### Budget semantics

`BudgetState.record_usage()` records a returned response without raising. Hard
token/cost checks happen before the next model call. This guarantees that edits,
validation, review, or finish calls already returned by the provider execute.
The finish tool also bypasses ordinary tool-call reservation, so a terminal call
cannot be lost to a reserve intended for future work.

The final budget snapshot is pushed back into the Controller before its summary
is rendered. `last_pass.model_calls_after` and `tokens_after` therefore include
the terminal response.

### Stable progress identity

Workspace progress is now a base-relative content fingerprint over meaningful
changed paths and file contents. It survives a commit with the same tree and
excludes `.forgeloop`, bytecode, coverage, pytest, mypy, and Ruff cache artifacts.
The same contract exists for local and Pier workspaces.

### Patch delivery

`RunDelivery` is a terminal boundary owned by `AgentLoop`. DeepSWE installs
`GitPatchDelivery`, which:

1. captures the official base SHA before the model starts;
2. removes only known ephemeral cache paths from delivery consideration;
3. stages the real working tree;
4. creates `forgeloop/deepswe-delivery-<base>` and commits real changes when
   needed;
5. verifies a non-empty `git diff --binary <base> HEAD`;
6. records `patch_delivery` with base/head SHA, commit status, clean status, and
   patch byte count.

A completed run with no deliverable patch becomes
`patch_delivery_failure`. Failed runs may still commit their genuine partial
work so the collector and trajectory retain evidence; no patch content is
invented and no verifier is bypassed.

## Removed behavior

The following older control paths were removed from the active v1.3 execution
path:

- mandatory edit-intent authorization;
- post-edit action gates and schema suppression;
- one-patch-per-response behavior;
- broad shell-as-validation inference;
- post-edit token reserve activation;
- scoped failure-read and post-review patch allowances;
- `no_progress_reinspect`, `no_progress_action_required`, and
  `exploration_action_blocked` forced-edit gates;
- pre-edit `implementation_due` tool blocking, which live trials showed reduced
  V4-Flash's ability to implement and was therefore removed.

Repeat-call, repeated-error, timeout, cancellation, safety, and hard overall
budget guards remain deterministic safeguards.

## Verification

### Deterministic integration coverage

`tests/test_execution_closure.py` executes a real Git repository through:

```text
two coherent source patches in one response
-> explicit pytest PASS
-> diff review
-> one unchanged decision
-> controller auto-finish
-> delivery branch and commit
-> non-empty base-to-HEAD patch containing both files
```

The test deliberately crosses the ordinary token maximum with an already
returned response and verifies exact PASS-to-terminal usage. Additional tests
cover validation failure/fix/retest, edits after review, validation that changes
the tree, finite phase exits, finish-over-budget execution, and fingerprint
stability across commit.

### Live acceptance on 2026-08-11

All DeepSWE runs used pinned upstream revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0, the official task and
verifier, and the unmodified V4-Flash policy. `model.patch` was inspected through
the real Pier job artifacts.

| Task / final run | Source diff | Agent validation | PASS -> terminal | Terminal | Collector patch | Official verifier |
|---|---:|---|---|---|---:|---|
| `query-encoding` / `20260811T022211Z-95bdb8e8` | yes | `python -m unittest tests.test_urls -v` PASS | 2 calls / 10,199 tokens | completed / `model_finish_tool` | internal eval: non-empty final diff (no Pier collector) | PASS, 3/3 |
| `sqlite-utils-safe-import-checkpoints` / `execution-closure-v2c-sqlite-20260811` | no | not reached | n/a | failed / `controller_exploration_exhausted`, 13 calls, 151,906 tokens | no | FAIL, F2P 0/60; P2P 1038/1038 |
| `textual-kitty-key-phases` / `execution-closure-v2f-textual-20260811` | no | not reached | n/a | failed / `provider_failure`, 8 calls, 45,163 tokens | no | FAIL, F2P 0/23; P2P 57/57 |
| `meriyah-explicit-resource-declarations` / `execution-closure-v2-meriyah-20260811` | no | not reached | n/a | failed / `provider_failure` on call 1 | no | FAIL, F2P 0/49; P2P 51469/51469 |
| `sqlfmt-create-table-ddl-formatting` / `execution-closure-v2-sqlfmt-20260811` | no | not reached | n/a | failed / `controller_exploration_exhausted`, 13 calls, 169,021 tokens | no | FAIL, F2P 0/32; P2P 1273/1273 |

Textual was retried twice after the final pre-edit hard gate was removed. The
provider disconnected without a response and then returned an SSL EOF. Meriyah
failed on the first provider request with the same external instability. These
are recorded as provider failures, not model or Controller successes.

Earlier design trials on Mashumaro, ABS, Cliffy, and Scriggo exposed that a
finite `implementation_due` schema gate made the model keep requesting hidden
reads instead of editing. That gate was removed before the final acceptance runs
above. Two Cliffy attempts were also invalidated by Docker Desktop shutting down;
Docker was restarted through its CLI and the infrastructure failures remain in
their original job directories.

### What the live results prove and do not prove

The query-encoding success proves the finite edit -> validation -> review ->
finish path and shows only two calls/10,199 tokens after PASS. Unit/integration
coverage proves that the same terminal path creates a clean, real base-to-HEAD
delivery commit and non-empty binary patch.

The live frozen DeepSWE samples did **not** produce a source edit under the final
policy samples, so they could not exercise validation or produce a non-empty
collector patch. The official verifiers therefore graded their pristine bases
and failed. This is not presented as a solve-rate improvement or as live proof
of a successful DeepSWE collector handoff. It is a remaining V4/provider
limitation: the harness now attributes and terminates it before a 200k
`budget_guard`, but cannot make the policy solve the code task.

## Quality gates

The release gate is:

```powershell
uv run pytest -q
uv run ruff check .
uv build
git diff --check
```

The build produces both `dist/forgeloop-0.1.0.tar.gz` and
`dist/forgeloop-0.1.0-py3-none-any.whl`.

## Remaining limitations

- No final live frozen DeepSWE sample reached source edit, so real non-empty Pier
  collection is verified by the delivery integration test rather than by a
  successful frozen trial.
- The V4-Flash endpoint showed repeated connection failures late in the run.
- The 75% exploration stop is a deterministic failure bound, not a claim that
  150k tokens is optimal for every repository. A returned response can cross the
  threshold before the next-call check.
- The policy/controller compatibility ID remains
  `deepseek-v4-flash-controller-v1.3-simplified`; the manifest now records
  `execution_closure: v2` so trajectories distinguish this architecture from the
  earlier v1.x patch stack.
