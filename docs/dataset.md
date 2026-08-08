# Trajectory to Dataset pipeline

ForgeLoop converts verifier-backed eval trajectories into a small, local JSONL
dataset. The pipeline is read-only with respect to source trajectories and keeps
the internal sample schema separate from training-framework adapters.

## Commands

```powershell
# Read .forgeloop/eval-runs and write the internal index and manifest.
uv run forgeloop dataset build

# Inspect classifications, source types, models, repositories, tokens, cost, and steps.
uv run forgeloop dataset inspect

# Export only sanitized sft_candidate records.
uv run forgeloop dataset export

# Export all non-infrastructure internal samples for later preference/RL work.
uv run forgeloop dataset export --format internal --output .forgeloop/dataset/curated.jsonl
```

`build` accepts repeatable `--suite` options when task provenance is stored in a
suite outside the bundled smoke suite or `.forgeloop/foundry/*/tasks.json`.
Source records without a readable trajectory are counted in the manifest as
`missing_trajectory`; the builder never invents a trajectory id.

## Internal sample schema

Each line of `index.jsonl` is a `forgeloop.dataset.sample.v1` object containing:

- task id, goal, mode, attempt, difficulty, expected outcome, and tags;
- repository, executable base SHA, suite/run/task provenance, upstream PR and
  commit metadata when available, and the source trajectory id and locator;
- model and provider;
- normalized conversation messages and tool schemas;
- ordered tool calls and observations;
- ordered structured effect events, an aggregate effect summary, and safety flags;
- final diff and Git status;
- verifier result and terminal/failure state;
- runtime metadata;
- steps, model/tool calls, token fields, cost, and wall time;
- classification and machine-readable classification reasons.

Fixture-only tasks use a stable `forgeloop-eval://<suite>/<task>` repository
identifier. Real-repository Foundry tasks retain their upstream repository, PR,
fix commit, and source base SHA. The executable fixture base SHA remains the
sample's top-level `base_sha`.

## Classification

- `sft_candidate`: the task result and verifier pass, infrastructure is healthy,
  and no strong inefficiency signal is present.
- `successful_but_inefficient`: verification passes but the run is blocked,
  budget-exhausted, no-progress, repeatedly calls the same tool, or repeats the
  same tool error.
- `model_failure`: the harness and environment are healthy, but the task result
  or verifier fails.
- `infrastructure_failure`: Harness, provider environment, Docker/runtime, or
  verifier infrastructure failed or no verifier result exists.

Infrastructure failures remain in the internal index for analysis, are excluded
from internal exports by default, and can never pass through the SFT adapter.
Original trajectories are never deleted or modified.

Effect fields are additive within `forgeloop.dataset.sample.v1`. A v1 trajectory
without Effect Events produces `effect_events=[]`, empty `safety_flags`, and an
`effect_summary` whose status is `legacy_no_effect_events`. New samples retain
the complete sanitized effects internally. The SFT adapter deliberately does not
insert Effect Events into model messages or otherwise expose them by default.

## SFT conversation adapter

The default export is `forgeloop.sft.conversation.v1` JSONL. Each record has:

```json
{
  "schema_version": "forgeloop.sft.conversation.v1",
  "id": "ds_...",
  "messages": [{"role": "system", "content": "..."}],
  "tools": [{"type": "function", "function": {"name": "..."}}],
  "metadata": {
    "task_id": "...",
    "repo": "...",
    "base_sha": "...",
    "trajectory_id": "...",
    "source_type": "...",
    "verifier_passed": true
  },
  "outcome": {"terminal_state": "completed", "final_diff": "...", "verifier": {}}
}
```

This is deliberately framework-neutral. A future trainer adapter can transform
the messages/tool arguments without changing the internal dataset schema.

## Sanitization

Sanitization runs both while building the index and immediately before export.
It removes exact credential values discovered from credential-like environment
variables, Authorization bearer values, API key/token/password assignments,
common provider token shapes, sensitive credential fields, repository-local
absolute roots, and user-home path prefixes. Relative repository paths and task
provenance remain available for training and audit.
