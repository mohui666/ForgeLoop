# Agent effects, replay, and deterministic explanation

ForgeLoop records structured side-effect evidence beside the existing model,
tool, observation, verifier, and terminal trajectory events. This observability
layer adapts the useful design lineage from TraceSeal at source commit
`7347c655e37109e6228dc743622f09248ccb1bab`: append-only ordered events,
file/Git before-and-after summaries, evidence-only replay, deterministic first
harmful-event analysis, and credential-safe HTTP evidence principles.

ForgeLoop does not import or execute TraceSeal. The implementation is native to
ForgeLoop's `ToolRegistry`, `Runtime`, `TrajectoryStore`, security, and Dataset
boundaries. It does not include TraceSeal's monkey-patch SDK, copied-workspace
sandbox, UI, Guard, Policy DSL, dashboard, or minimizer.

## Effect schema

Each `effect` trajectory event contains a `forgeloop.effect.v1` payload:

```json
{
  "schema_version": "forgeloop.effect.v1",
  "event_id": "eff_abc12345_0001",
  "trajectory_id": "abc12345...",
  "step": 3,
  "timestamp": "2026-08-08T12:00:00+00:00",
  "type": "file.write",
  "tool_name": "apply_patch",
  "tool_call_id": "call-3",
  "target": "src/example.py",
  "action": {"operation": "update", "replacement_count": 1},
  "result": {"status": "success"},
  "risk": {"level": "low", "flags": []},
  "evidence": {
    "before": {"size": 20, "sha256": "..."},
    "after": {"size": 24, "sha256": "..."}
  }
}
```

Supported types are `file.read`, `file.write`, `file.delete`, `shell.exec`,
`git.change`, `test.run`, `http.request`, and `policy.violation`.
`http.request` is reserved for a future ForgeLoop Runtime/tool that performs
HTTP; the current Runtime has no HTTP operation, so ForgeLoop does not infer
network events from arbitrary shell text. `policy.violation` currently records
existing shell and sensitive-path safety rejections without introducing a new
Policy Engine.

One tool call can emit zero or more effects. `apply_patch`, for example, can emit
both `file.write` and `git.change`; a test shell can emit `shell.exec` and
`test.run`; a shell deletion visible to Git can additionally emit `file.delete`
and `git.change`. Every emitted effect retains the Agent step and tool-call id.

Evidence stores file size/hash rather than source content, workspace-relative
targets, compact Git before/after state, command exit state, and bounded
stdout/stderr excerpts. The shared ForgeLoop security sanitizer removes exact
environment credentials, API keys, tokens, Authorization/cookie headers,
provider-token shapes, URL userinfo, sensitive query assignments, workspace
roots, and user-home prefixes before persistence.

## Offline replay

```powershell
uv run forgeloop trace replay <trajectory-id-or-jsonl-path>
```

Replay reads JSONL only. It never invokes a Runtime, tool, shell, file writer, or
network client. The compact timeline includes the task/model, tool calls,
important observations, effects, verifier, and terminal state in recorded order.

## Deterministic explain

```powershell
uv run forgeloop trace explain <trajectory-id-or-jsonl-path>
```

Explain uses no model. It reports only evidence-supported facts:

- final verifier and terminal state;
- affected files and recorded test results;
- repeated effects/tool calls and repeated failed observations;
- effects after a PASS verifier;
- destructive/sensitive effects and safety flags;
- the final effect before `no_progress`;
- the earliest supported suspicious, unnecessary, or harmful effect.

When evidence is absent, the command prints `UNKNOWN`, `none recorded`, or
`None supported by recorded evidence` instead of inventing a cause.

## Compatibility

New trajectories use `forgeloop.trajectory.v2`. Replay and Dataset parsing still
accept `forgeloop.trajectory.v1`. Dataset samples keep the additive
`forgeloop.dataset.sample.v1` contract and now include `effect_events`,
`effect_summary`, and `safety_flags`. Old trajectories and old dataset indexes
load with empty effects and `effect_summary.status=legacy_no_effect_events`;
ForgeLoop never fabricates historical effects.
