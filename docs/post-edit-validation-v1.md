# Post-Edit Validation v1

Post-Edit Validation v1 adds a narrow deterministic validation loop to the
existing `deepseek-v4-flash-controller-v1.3-simplified` policy. It preserves
V4-Flash, Agent Context Efficiency v1, the advisory
`qwen2.5:1.5b-instruct` classifier, all tool schemas, and the official DeepSWE
tasks, collector, and verifier. It does not restore Edit Intent or add a
supervisor or classifier-driven phase gate.

## Behavior

The Controller uses the existing Git progress fingerprint and tool results:

- after a real source diff, one focused source read is still available, but
  broad listing/search and another patch are blocked until V4-Flash runs a
  repository-appropriate test or executable validation command;
- if no known test command exists, V4-Flash selects the smallest relevant
  Python, JavaScript, type, lint, build, or repository test command;
- a failed validation permits a focused failure read, direct patch, and retest;
- a passing validation requires a current diff review before completion;
- after review, commit/finish is preferred; a direct patch remains available
  for a concrete problem found in the diff and invalidates the passing result,
  requiring another validation;
- a command presented as validation cannot validate a source state that it also
  changes. The changed fingerprint becomes the new unvalidated state;
- before any source diff, broad exploration stops when total usage reaches 75%
  of `max_tokens`, preserving approximately 25% for edit, validation, repair,
  diff review, and finalization.

This is deterministic control over evidence already available to `AgentLoop`.
The 1.5B classifier remains advisory telemetry and never authorizes an action.

Trajectories add `controller_post_edit_validation`,
`controller_post_edit_diff_reviewed`, `controller_post_edit_action_blocked`,
and `controller_validation_reserve_activated` events. The final Controller
summary records attempts, passes, failures, source-changing validation calls,
retests, diff review, blocked reasons, and reserve activation. Existing
`context_usage` events continue to expose actual per-call input tokens,
compaction, and dominant context sources.

## One-shot validation on 2026-08-11

The frozen internal `query-encoding` task ran once and passed. Both requested
DeepSWE tasks ran once through pinned revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0, V4-Flash, Controller
v1.3 Simplified, Agent Context Efficiency v1, and the unchanged official
collector/verifier. Failed runs were not retried.

| Task | Input tokens | Model / tool calls | Observed path | Terminal | Verifier |
|---|---:|---:|---|---|---:|
| `query-encoding` | 14,560 | 6 / 8 | edit -> validation pass -> diff review -> finish | completed / `model_finish_tool` | PASS |
| `sqlite-utils-safe-import-checkpoints` | 187,617 | 17 / 19 | edit -> validation pass -> diff review; no finish | budget exceeded / `budget_guard` | FAIL |
| `textual-kitty-key-phases` | 175,613 | 14 / 18 | edit -> validation fail -> retest pass -> diff review; no finish | budget exceeded / `budget_guard` | FAIL |

Compared with the immediately preceding Agent Context Efficiency v1 one-shot
runs, input changed from 15,398 to 14,560 for `query-encoding`, 184,983 to
187,617 for `sqlite-utils`, and 184,943 to 175,613 for `textual`. The two
DeepSWE samples together used 363,230 input tokens instead of 369,926 (-1.8%)
and 31 model calls instead of 33. This small sample does not establish a token
or solve-rate improvement; its purpose is to verify the post-edit execution
path.

Both DeepSWE trajectories satisfy the central execution invariant: after the
first source diff, the agent actually attempted validation instead of
continuing pure exploration until the budget guard. Neither solved. SQLite
used an import-only check, then later issued a Python command that modified the
source while appearing to validate it. That observation led to the final
source-fingerprint invalidation rule. Textual's first check failed and its
focused retest passed, but the patch only changed `src/textual/keys.py`, was
incomplete for the official 23-test requirement, and was never committed; the
official collector therefore graded the pristine base and reported 0/23 F2P,
57/57 P2P, partial 0.7125.

The final post-review direct-patch allowance and source-changing-validation
invalidation were tightened from these one-shot trajectories and covered by
deterministic unit tests; the frozen tasks were not rerun to avoid replacing
the required single samples. The final code therefore has no additional live
solve claim.

Per-call context evidence remained present. The maximum provider input was
3,210 tokens for `query-encoding`, 24,035 for `sqlite-utils`, and 26,567 for
`textual`; context compaction applied on 13/17 and 10/14 DeepSWE calls,
respectively.

## Verification

The repository validation commands are:

```powershell
uv run pytest
uv run ruff check .
uv build
```
