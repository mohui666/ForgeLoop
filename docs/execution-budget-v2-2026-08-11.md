# Execution Budget v2

Date: 2026-08-11

Implementation base: `e4431ae259b0fab5c644eebe549a8c77357eb3ff`

Policy under test: `deepseek-v4-flash-controller-v1.3-simplified`

DeepSWE checkout: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`

## Result

Execution Budget v2 removes cumulative provider token usage from ForgeLoop's
execution authorization. Input, cached, reasoning, output, total-token, cost,
model, and provider accounting remain unchanged telemetry. Another model call
is now governed by model/step calls, wall-clock time, tool-call safety, optional
cost safety, and provider/output limits. The frozen provider generation limit
remains 8,192 output tokens per response.

DeepSWE now defaults to a configurable long-horizon window of 256 model calls,
256 steps, 1,024 tool calls, and 5,400 seconds. Pier receives these values as
explicit agent kwargs. Every mapped run records
`forgeloop.execution-budget.v2`, `cumulative_tokens: telemetry_only`, and the
resolved limits in `provenance.json`.

The live validation proves the old 150k/200k termination is gone. All three
tasks continued past 200k cumulative input; final input ranged from 813,621 to
2,613,643 tokens and trajectories reached 47 to 100 model calls. None stopped
with `budget_guard` or `controller_exploration_exhausted`.

It does not show a solve-rate improvement. ABS and SQLfmt remained in
exploration until the unchanged repeated-tool safety guard fired. SQLite first
edited at call 98, then the frozen post-edit controller ended the run before a
validation action. No task finished and the official verifier failed all
three.

## Frozen scope

The system prompt, native tools, Controller v1.3 implementation, Context
Efficiency v1, V4-Flash policy manifest, 1.5B advisory classifier, task images,
and verifiers were unchanged. Pier remains pinned to 0.3.0; the known v1.1
collector/version issue was not repaired. No SFT or RL work was performed.

## Canary

The unchanged internal `query-encoding` canary passed once after refreshing the
DeepSeek credential inherited by the evaluation process.

| Result | Calls / tools | Input / cached / output | Wall time | Cost | Termination |
|---|---:|---:|---:|---:|---|
| PASS | 6 / 8 | 16,915 / unavailable / 932 | 32.779s | $0.00074997 | `model_finish_tool` |

Artifact:
`.forgeloop/execution-budget-v2-canary-valid/20260811T130137Z-4d9a9266`.

Two earlier canary attempts were retained as infrastructure preflights. Both
failed on the first provider request with `Insufficient Balance`, performed no
tool action, and have unavailable usage. They are not counted as model
behavior attempts.

## DeepSWE one-shot results

Call positions are successful model responses. Cumulative usage is measured at
the response that emitted the action.

| Task | Result | Calls / tools | First behavioral test or probe | First source edit | Edit / test / finish | Termination |
|---|---|---:|---|---|---|---|
| ABS | FAIL | 47 / 68 | none | none | no / no / no | `repeated_tool_call` |
| SQLfmt | FAIL | 57 / 73 | formatter probe: call 23, 258,224 / 63,232 / 3,461; first pytest: call 28, 319,872 / 86,912 / 4,649, PASS (18 tests) | none | no / yes / no | `repeated_tool_call` |
| SQLite | FAIL | 100 / 135 | SQLite transaction probe: call 52, 885,263 / 343,424 / 32,520; no test suite | call 98, `sqlite_utils/db.py`, 2,528,619 / 1,322,240 / 129,254 | yes / no / no | `controller_validation_not_reached` |

Usage in the action columns is input / cached / output.

| Task | Final input / cached / output | Wall time | Cost | Delivery / collector | Official verifier |
|---|---:|---:|---:|---|---|
| ABS | 813,621 / 315,136 / 63,877 | 848.393s | $0.08855584 | no source change; no patch | FAIL |
| SQLfmt | 970,735 / 350,080 / 15,747 | 685.373s | $0.09228108 | no source change; no patch | FAIL |
| SQLite | 2,613,643 / 1,385,472 / 131,074 | 1,819.673s | $0.21252398 | ForgeLoop delivery committed a 7,570-byte patch; Pier 0.3.0 did not collect `model.patch` | FAIL |

Artifacts:

- `.forgeloop/eval-v2-runs/execution-budget-v2-abs-valid-20260811`
- `.forgeloop/eval-v2-runs/execution-budget-v2-sqlfmt-valid-20260811`
- `.forgeloop/eval-v2-runs/execution-budget-v2-sqlite-valid-20260811`

Three earlier DeepSWE runs were also retained as infrastructure preflights.
Each rejected its first provider request with `Insufficient Balance`; their
usage and cost are recorded as unavailable rather than zero.

## Accounting and regression evidence

`BudgetState.record_usage` still adds every reported input, cached, reasoning,
output, and cost field without subtracting cache hits. When a provider request
fails before returning usage, mapped DeepSWE reports now preserve those fields
as unavailable rather than converting unknown values to zero.

The focused regression test records two complete responses totaling 250,010
input, 200,000 cached, and 10 output tokens. The second response executes a
normal `finish` after the first response alone exceeded the historical 200k
limit. Focused budget/DeepSWE/agent/eval tests passed before the live runs.

## Validation

Final validation commands:

```powershell
uv run pytest
uv run ruff check .
uv build
git diff --check
```

Pytest passed with 145 tests. Ruff passed and `uv build` produced both the
source distribution and wheel.

## Conclusion

Execution Budget v2 meets its semantic objective: cumulative context replay is
usage/cost telemetry, not a pre-edit hard stop, and DeepSWE has an explicit,
configurable long-horizon execution window. The honest one-shot result is 0/3
solved. Removing the bad token horizon enabled materially longer execution and
one late source edit, but it did not by itself make the frozen agent reliably
converge, validate, or finish.
