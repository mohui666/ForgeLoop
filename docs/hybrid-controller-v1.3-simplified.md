# Controller v1.3 Simplified

> Policy compatibility report. The active execution semantics are documented in
> [Execution Closure v2](execution-closure-v2.md).

Controller v1.3 Simplified removes the mandatory Edit Intent Handoff and the
classifier-driven phase gate. The bundled policy is
`deepseek-v4-flash-controller-v1.3-simplified`.

The earlier v1.1, v1.2, Edit Intent, and readiness policies remain bundled for
historical provenance. This version does not change V4-Flash, AgentLoop, tool
schemas, DeepSWE tasks, Docker environments, patch collection, or verifiers.

## Behavior

Local `qwen2.5:1.5b-instruct` still classifies the compact trajectory state as
`explore`, `implement`, `verify`, or `finalize`, but its decision is telemetry
only:

- it does not filter the tools exposed to V4-Flash;
- it does not authorize or reject inspect, edit, or test actions;
- malformed or unavailable classifier output falls back without blocking;
- no `submit_edit_intent` tool is registered or required.

V4-Flash normally receives the complete ForgeLoop tool set throughout the run.
The existing deterministic Controller still handles budget, timeout, provider
and tool errors, repeated actions, prolonged no-progress, Git progress, and
terminal safety. Existing tool/runtime security checks remain authoritative.

When a real Git-visible source change appears, Controller v1.3 emits one fixed
guidance message for that new fingerprint: run a focused test, fix its concrete
failure if needed, then call `finish`. [Execution Closure v2](execution-closure-v2.md)
now adds narrow deterministic evidence requirements around that path. It keeps
all tool schemas visible and the classifier advisory, while blocking broad
post-edit exploration until validation and requiring a passing current diff
before `finish(completed)`.

## Run

```powershell
$env:DEEPSEEK_API_KEY = "..."
ollama pull qwen2.5:1.5b-instruct
uv run forgeloop controller probe

uv run forgeloop task "Fix the failing test" `
  --policy-manifest deepseek-v4-flash-controller-v1.3-simplified
```

## One-shot validation on 2026-08-09

### Frozen `query-encoding`

The unchanged frozen task passed its verifier and ended
`completed/model_finish_tool`:

- 7 model calls and 9 tool calls;
- 17,414 input and 1,342 output tokens, 18,756 total;
- 29.86 seconds and $0.00077657 API cost;
- actual sequence: inspect -> source edit -> test -> diff -> explicit finish;
- no edit-intent event or classifier action block occurred;
- all classifier decisions were recorded with `advisory=true`.

### Three frozen DeepSWE tasks

Three tasks that previously ended at `controller_invalid_edit_intent` were each
run once with pinned DeepSWE revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0, and their unchanged
official environments and verifiers. Failures were not retried.

| Task | Edit | Agent validation | Terminal | Tokens | Verifier |
|---|---:|---:|---|---:|---:|
| `sqlfmt-create-table-ddl-formatting` | no | no | failed / `controller_no_change_final` | 162,514 | FAIL |
| `sqlite-utils-safe-import-checkpoints` | yes | yes | budget exceeded / `budget_guard` | 216,068 | FAIL |
| `textual-kitty-key-phases` | yes | no | budget exceeded / `budget_guard` | 201,060 | FAIL |

`sqlite-utils` made six successful source edits and ran a concrete Python
checkpoint validation script. The script exposed an error; V4 continued
inspecting and patching until the token guard stopped the run. `textual` made a
Git-visible source edit but did not reach a test. `sqlfmt` remained in prolonged
exploration and ended after the deterministic no-progress recovery without a
change.

None of the three runs emitted an edit-intent event or ended for an invalid
intent. Two of three reached source editing and one reached real validation, so
the artificial intent bottleneck is removed. None solved the task or produced a
DeepSWE collector patch. The remaining failure is primarily V4 exploration and
token efficiency, not a new Controller gate. No solve-rate improvement claim is
made from these three samples.
