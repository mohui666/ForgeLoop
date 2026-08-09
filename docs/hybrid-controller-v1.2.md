# Hybrid Controller v1.2: state-aware tool gating

Hybrid Controller v1.2 keeps the v1.1 architecture and makes the Controller
state constrain the tools exposed to V4-Flash on the next model call. It does
not change `AgentLoop`'s tool protocol, the DeepSWE patch collector or verifier,
and it does not add planning, reflection, SFT, RL, or another Agent.

The bundled policy is
`deepseek-v4-flash-hybrid-controller-v1.2`. V4-Flash remains the only coding
model. The local `qwen2.5:1.5b-instruct` Ollama model only returns a validated
enum pair:

```text
state = explore | implement | verify | finalize
next_action = inspect | edit | test | replan | finalize
```

Deterministic Controller v1 remains authoritative for budget, timeout,
provider/tool errors, repeated actions, Git progress and terminal decisions.

## State gates

The Controller filters the existing tool schemas before every provider call
and validates every requested action again before execution:

| State | Intended action space |
| --- | --- |
| `explore` | Normal inspect, list and search tools. |
| `implement` | Explicit file reads, patch/edit, diff and bounded test/format commands. Repository-wide list/search and broad shell discovery are blocked. After three successful scoped reads in one implement epoch, further reads are removed so the next choice must make progress or terminate honestly. |
| `verify` | Tests, failed-output inspection, scoped reads, patch repair and diff. Broad discovery is blocked. |
| `finalize` | Diff, test/format, final patch, Git finalization and explicit `finish`. Broad discovery is blocked. |

The schema filter is backed by an execution-time guard because a provider can
return multiple tool calls based on the schema snapshot from the beginning of
the turn. A rejected call is never executed. It records a
`controller_action_blocked` event containing the state, rejected action,
reason and the currently allowed action names, then returns fixed feedback to
V4-Flash.

## Escape hatch

The first blocked list/search request in a state epoch exposes one controlled
targeted replan. Broad shell payloads never receive this escape. If V4-Flash uses it, ForgeLoop records
`controller_replan_allowed`, closes the escape immediately and returns to the
restricted action space. A genuine classifier transition from a later phase
back to `explore` may reset the deterministic no-progress inspection window
once for the entire run. That reset is recorded as
`controller_replan_window_reset`; it cannot repeat indefinitely.

Every run summary includes blocked counts by state, controlled replans, the
one-time deterministic reset, scoped-read allowance and located paths.

## Run

```powershell
$env:DEEPSEEK_API_KEY = "..."
ollama pull qwen2.5:1.5b-instruct
uv run forgeloop controller probe
uv run forgeloop task "Fix the failing test" `
  --policy-manifest deepseek-v4-flash-hybrid-controller-v1.2
```

The canonical V4-Flash interactive route selects v1.2. Historical v1 and v1.1
manifests remain bundled for provenance and comparisons.

## Real validation on 2026-08-09

### Frozen `query-encoding`

The final unchanged LocalRuntime task passed its verifier and ended
`completed/model_finish_tool`:

- 6 steps/model calls, 8 tool calls;
- 15,175 input and 1,201 output tokens;
- 23.668 seconds, provider cost $0.00109098;
- local Controller: 6 decisions, 991 input and 108 output tokens, 2.720205
  seconds, $0, no fallback;
- transition path: `explore -> implement -> verify -> finalize`;
- real tool path: search/list, two scoped reads, `apply_patch`, test, diff,
  `finish`.

One implement-state action was blocked. The task then edited, tested and
finished correctly, so v1.2 does not regress the frozen smoke task.

### DeepSWE `oxvg-structural-selector-preservation`

The scoped-read gating candidate was run once through pinned DeepSWE revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, the official Pier 0.3.0 Docker
environment, unchanged patch collector and official verifier. The result is a
real failure:

- terminal: `budget_exceeded/budget_guard`;
- 16 steps/model calls, 19 tool calls;
- 187,702 input and 12,347 output tokens, 248.730 seconds, $0.007801;
- no patch collected;
- verifier: P2P 62/62, F2P 0/6, partial 0.9117647058823529, reward 0;
- attribution: `model_failure`, not provider, Docker, collector or verifier
  failure.

The Controller made 8 validated decisions with no fallback. It entered
`implement`, blocked 8 disallowed actions, allowed one controlled replan and
stopped source reads after the three-read allowance. V4-Flash nevertheless
kept selecting blocked search/read/shell actions and Git inspection instead of
`apply_patch`; it never reached a test action. This validates that unbounded
exploration is now mechanically gated, but it does **not** demonstrate that
V4-Flash will always comply by editing. The task remains unsolved and no solve
or self-improvement claim is made.

The post-run event audit found that `git_inspect(log)` was still argument-level
history exploration even though its tool name was allowed for status/diff.
Final v1.2 blocks that operation and mixed shell payloads such as
`test && find`; both cases have regression tests. This narrow audit fix was not
followed by another paid model run.

Two earlier development attempts are retained as local evidence. They exposed
and led to fixes for an over-restrictive located-file scope and a stale
deterministic no-progress window after classifier replan. They are not counted
as successful evaluations.
