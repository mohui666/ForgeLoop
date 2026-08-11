# Agent Policy v1: Execution-first coding behavior

Agent Policy v1 is a prompt-only attempt to move the main coding model from
repository analysis into an executable edit/validation loop sooner. Controller
v1.3 Simplified, Execution Closure v2, Agent Context Efficiency v1, the DeepSWE
tasks, Pier collector, and official verifiers were held fixed.

The intended behavior is:

```text
inspect -> hypothesis -> minimal edit -> validate -> fix/retest -> finish
```

This release records a negative frozen-task result honestly: the policy kept the
`query-encoding` canary passing, but V4-Flash still produced no source edit in
all four selected DeepSWE samples. No second prompt stack was added after that
result.

## Pre-change audit

The previous system prompt contained two open-ended instructions:

- "Use tools to inspect facts before changing code."
- Goal Mode began with "Analyze the repository" and forming a plan.

It required validation after editing, but described validation mainly as proof
needed before completion. It did not tell the model to use a focused failing
test as an experiment, define when local evidence was sufficient, or distinguish
a targeted read from an attempt to understand the whole repository.

Recent final-architecture trajectories showed the resulting behavior:

| Task | Model calls | Tool calls | Tokens | First source edit | Terminal |
|---|---:|---:|---:|---|---|
| `sqlite-utils-safe-import-checkpoints` | 13 | 18 | 151,906 | none | `controller_exploration_exhausted` |
| `sqlfmt-create-table-ddl-formatting` | 13 | 27 | 169,021 | none | `controller_exploration_exhausted` |

Both runs began by saying they would explore the repository to understand what
they were working with. SQLite then read increasingly large ranges of
`sqlite_utils/db.py` and `sqlite_utils/cli.py`. SQLfmt expanded from rule files
into the parser, line, merger, node, token, query, formatter, analyzer, and mode
modules. The Controller's advisory that source evidence was ready did not change
either run into an edit. Neither run invoked validation or finish.

This evidence supported a prompt-level hypothesis: the model needed an explicit
local-evidence stopping condition and permission to use a minimal patch plus
focused validation to resolve remaining uncertainty.

## Implementation

Editing modes now receive one versioned `execution-first-v1` policy block. It:

- directs the model to localize through task terms, observed failures, nearby
  tests, and direct callers;
- says not to wait for complete repository understanding;
- prefers the smallest coherent, reversible patch once local evidence supports
  a plausible hypothesis;
- treats focused validation as an information-gathering experiment;
- rejects broad surveys, exhaustive dependency reading, repeated searches, and
  unchanged rereads unless a specific answer would alter the next edit or test;
- makes final diff review and finish the immediate path after PASS.

Plan Mode does not receive editing instructions. The bundled V4-Flash manifest
records `serving_config.agent_policy = "execution-first-v1"`, so trajectories
identify the prompt policy without changing the frozen model or Controller
identity.

No Controller code, state transition, budget, planner, intent handoff,
supervisor, DeepSWE task, verifier, collector, training path, or model selection
was changed.

## Live acceptance

Runs used DeepSWE revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0, V4-Flash, and the
unchanged `deepseek-v4-flash-controller-v1.3-simplified` policy identity. Each
task was run once. Inspection of every first `model_request` confirmed the
execution-first policy text was present.

| Task / run | First source edit | Source diff | Agent validation | Finish | Collector `model.patch` | Verifier result |
|---|---|---:|---|---:|---:|---|
| `query-encoding` / `20260811T100003Z-4fac55a7` | call 3 / 9,174 tokens | yes, `urls.py` | call 4 / 13,988 cumulative tokens, PASS | yes, call 6 / `model_finish_tool` | n/a; internal eval retained a non-empty final diff | PASS, 3/3 |
| `abs-stepped-slices` / `agent-policy-v1-abs-20260811` | none; 16 calls / 153,885 total | no | not reached | no | no file; `no_patch_collected` | FAIL, F2P 0/6; P2P 6/6 |
| `mashumaro-flattened-dataclass-fields` / `agent-policy-v1-mashumaro-20260811` | none; 12 calls / 170,699 total | no | not reached | no | no file; `no_patch_collected` | FAIL, F2P 0/66; P2P 30014/30014 |
| `sqlfmt-create-table-ddl-formatting` / `agent-policy-v1-sqlfmt-20260811` | none; 15 calls / 165,443 total | no | not reached | no | no file; `no_patch_collected` | FAIL, F2P 0/32; P2P 1273/1273 |
| `sqlite-utils-safe-import-checkpoints` / `agent-policy-v1-sqlite-20260811` | none; 13 calls / 155,924 total | no | not reached | no | no file; `no_patch_collected` | FAIL, F2P 0/60; P2P 1038/1038 |

For `query-encoding`, PASS occurred on model call 4. The final diff review and
finish used two additional model calls and 10,462 tokens. This preserves the
known successful execution/closure path.

All four Pier job trees were inspected recursively: none contained a
`model.patch`. The official verifiers therefore graded the pristine base state.
There is no live frozen-task collector success to claim for Agent Policy v1.

## Comparison with prior 0-edit samples

ABS and Mashumaro are earlier failed design-trial references; SQLfmt and SQLite
are the exact post-Closure-v2 baselines. All comparisons use one recorded
attempt and the same no-edit outcome.

| Task | Prior tokens | Policy v1 tokens | Change | Policy v1 edit |
|---|---:|---:|---:|---:|
| ABS | 152,829 | 153,885 | +0.7% | no |
| Mashumaro | 145,767 | 170,699 | +17.1% | no |
| SQLfmt | 169,021 | 165,443 | -2.1% | no |
| SQLite | 151,906 | 155,924 | +2.6% | no |
| Mean | 154,881 | 161,488 | +4.3% | 0/4 |

The target of clearly reducing `100k+ tokens / 0 edit` was not met. The small
SQLfmt reduction is noise relative to the unchanged outcome, while average use
increased. In all four policy-v1 runs the first reasoning message still said it
would explore the repository to understand what it was working with. Later
calls continued reading even when the execution-first system text and
Controller convergence advisory were both present.

## Decision and next step

The prompt now states the desired coding behavior directly and removes the
previously ambiguous full-understanding cues. The live result nevertheless
falsifies the hypothesis that prompt wording alone can make this V4-Flash policy
reliably enter execution on hard frozen tasks. Adding more reminders, numeric
read limits, or forced-edit language would recreate the prompt/guard stack that
earlier trials already showed to be brittle.

The next experiment should keep the AgentLoop, Controller, Closure, Context,
tasks, budgets, and this system prompt fixed, then compare coding-specialized
main-model candidates once on the same canary and four frozen tasks. Select on
first-edit reliability and end-to-end delivery, not only verifier score:

1. first source-edit call and cumulative tokens;
2. proportion of tasks producing a real source diff before 100k tokens;
3. entry into focused validation and repair/retest;
4. explicit finish and non-empty collector patch;
5. official F2P/P2P result and regression preservation.

A practical promotion threshold is at least three of four frozen tasks producing
a source edit, with a materially lower median first-edit cost, before spending
more work on prompt policy for V4-Flash.

## Quality gate

The release gate is:

```powershell
uv run pytest -q
uv run ruff check .
uv build
git diff --check
```

All four commands passed on 2026-08-11. Ruff reported no violations. The build
produced both `dist/forgeloop-0.1.0.tar.gz` and
`dist/forgeloop-0.1.0-py3-none-any.whl`.
