# DeepSeek V4-Flash + Controller v1

ForgeLoop exposes DeepSeek V4-Flash as the bundled policy
`deepseek-v4-flash-controller-v1`. It uses the official OpenAI-compatible
Chat Completions route at `https://api.deepseek.com/v1`, model
`deepseek-v4-flash`, thinking level `max`, and the existing ForgeLoop tool
schemas. The policy manifest contains only the credential environment-variable
name; it never stores the credential itself.

```powershell
$env:DEEPSEEK_API_KEY = "..."
uv run forgeloop policy probe --policy deepseek-v4-flash-controller-v1
uv run forgeloop task "Fix the failing test" `
  --policy-manifest deepseek-v4-flash-controller-v1
```

The interactive TUI also binds this policy and Controller automatically when
the selected canonical model is `deepseek/deepseek-v4-flash`. The provider
adapter preserves `reasoning_content` on thinking-mode tool-call turns, as
required by DeepSeek's multi-turn tool protocol. No `tool_choice` is sent.

The manifest records the public API release identity (`2026-04-24`), not an
immutable weight digest: DeepSeek controls the hosted model alias. Capability
sources and the deployment identity are retained in every trajectory.

## Controller v1

Controller v1 is deterministic code around the existing `AgentLoop`; it does
not add a planner, critic, reflection pass, model call, tool, or alternate tool
protocol. It is enabled only when a policy manifest declares
`serving_config.controller = "v1"`, so the existing Qwen policies and frozen
eval configuration remain unchanged.

Every recovery is appended as `controller_recovery` with a strategy, trigger,
step, and the exact feedback returned to the model. `run_finished` includes the
per-strategy counts. The implemented strategies are:

- `repeated_action`: flag the second identical tool call before the legacy hard
  repeat limit.
- `tool_error_feedback`: explicitly direct the model to the authoritative raw
  tool error already present in history.
- `edit_failure_reinspect`: after two failed `apply_patch` attempts, require a
  fresh read and a smaller patch.
- `no_progress_reinspect`: after six actions without Git-visible progress,
  request a focused re-inspection and edit.
- `no_progress_action_required`: allow two focused inspections, then require an
  edit or explicit blocked/failed finish.
- `exploration_action_blocked`: while action-required is active and no change
  exists, reject further read/search/shell exploration as a normal tool error;
  `apply_patch` and `finish` remain available.
- `missing_explicit_finish`, `exploration_without_change`, and
  `finish_without_change`: give one deterministic recovery opportunity before
  an explicit Controller terminal reason.

Budget and time limits remain in `AgentLoop`. A wall-clock budget ends as
`timeout_guard`; provider timeouts and provider failures end as
`provider_timeout` and `provider_failure`; other budgets remain `budget_guard`.
Controller recovery never retries a failed provider call implicitly.

Two harness issues exposed during validation were fixed with regression tests:

1. `.forgeloop/` trajectory files appeared in `git status` and falsely counted
   as repository progress. They are now excluded from the progress fingerprint.
2. The legacy shell no-progress counter terminated on the same turn as the
   first Controller recovery. With Controller enabled, that first window is now
   reset so the model can receive the recovery; Controller's deterministic
   action gate and the outer step/token/time budgets remain hard boundaries.

## Real validation on 2026-08-09

The live `/v1` probe returned a valid `forgeloop_probe` tool call with the exact
nonce against `https://api.deepseek.com/v1`: 388 input tokens, 56 output
tokens, `finish_reason=tool_calls`, and 1.323 seconds provider latency.

The unchanged frozen `query-encoding` task passed its verifier on the first
attempt. V4-Flash used 9 model calls, 12 tool calls, 25,213 input tokens, 1,731
output tokens, and 24.992 seconds. It produced a non-empty `urls.py` patch,
passed all 3 verifier tests, and terminated `completed/model_finish_tool`.
Controller actually triggered `tool_error_feedback` once and
`repeated_action` three times.

DeepSWE used the pinned Eval v2 revision and official Docker/Pier verifier. The
initial `katex-multicolumn-array-spans` attempt terminated
`budget_exceeded/budget_guard` after 10 model calls and 217,769 input tokens;
it made no edit, so Pier recorded `no_patch_collected`. The official verifier
kept 599/599 existing tests passing and passed 0/94 feature tests.

`oxvg-structural-selector-preservation` was used to regression-test the
Controller fixes. The final enforced-gate run triggered
`no_progress_reinspect`, `no_progress_action_required`, five
`exploration_action_blocked` recoveries, and later a second no-progress cycle.
The model did respond with one successful `apply_patch`, creating an uncommitted
WIP file, but continued exploring instead of committing. The 200,000-token
budget then terminated the Agent after 15 model calls, 22 tool calls, 184,946
input tokens, and 20,216 output tokens. Because DeepSWE collects committed
changes, Pier correctly recorded `no_patch_collected`; its verifier kept 62/62
existing tests passing and passed 0/6 feature tests.

Therefore the V4-Flash route, tool protocol, Controller recovery events,
terminal attribution, frozen-task patch flow, and official DeepSWE verifier
flow are live. A non-empty **committed** DeepSWE patch was not achieved in this
validation, and no claim to the contrary is made. The observed remaining model
failure is over-exploration and failure to commit before the token budget;
Controller v1 makes this attributable and bounded but does not solve the task
for the model.

Official references:

- [DeepSeek V4 release](https://api-docs.deepseek.com/news/news260424/)
- [DeepSeek model details and pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [DeepSeek thinking-mode history requirements](https://api-docs.deepseek.com/guides/thinking_mode)
- [DeepSeek tool calls](https://api-docs.deepseek.com/guides/tool_calls)
