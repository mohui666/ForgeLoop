# DeepSWE Eval v2: V4-Flash + implement readiness (20 tasks)

## Frozen configuration

- Run date: 2026-08-09
- ForgeLoop commit under evaluation:
  `024083241dd4c6be3319e24c97748a4d0bf1cc0a`
- Policy: `deepseek-v4-flash-edit-intent-readiness-v1`
- DeepSWE revision: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`
- Subset: `deepswe-eval-v2-daily-20`
- Sampling seed: `20260809`
- Pier: 0.3.0
- Docker server: 29.6.2
- Job: `deepswe-eval-v2-daily-20-20260809T105055Z`

All 20 frozen tasks were run once. Controller logic, prompts, models, task
inputs, Docker environments, patch collector, and official verifiers were not
changed during or after the run. Failures were not retried.

Command:

```powershell
uv run forgeloop deepswe run `
  --policy deepseek-v4-flash-edit-intent-readiness-v1
```

## Aggregate result

| Metric | Result |
|---|---:|
| Solved | 0 / 20 |
| Valid edit intents | 1 |
| Tasks with Git-visible source changes | 2 |
| Final patches collected by DeepSWE | 0 |
| Tasks that actually ran an Agent test command | 0 |
| Explicit `finish` calls | 0 |
| Budget exceeded | 2 |
| Failed terminal | 18 |
| Total input tokens | 1,548,505 |
| Total output tokens | 117,696 |
| Total tokens | 1,666,201 |
| Batch wall time | 48m 13s |
| Sum of task wall times | 2,893.679s |
| API cost | $0.14869613 |

All 20 failures were classified as `model_failure`. Terminal and stop-reason
counts were:

- `failed / controller_invalid_edit_intent`: 18
- `budget_exceeded / budget_guard`: 2

“Git-visible source changes” means the terminal trajectory recorded a dirty
source path. `cliffy-config-file-parsing` created three untracked TypeScript
files after its accepted intent. `scriggo-method-declarations` modified
`internal/compiler/parser_func.go` without first completing the intent handoff.
Neither task produced a final collector artifact, so the stricter DeepSWE
`model.patch` count is zero.

“Entered test” means V4 actually executed a shell command recognized as a test,
not merely that the classifier selected `verify`, an intent contained a
validation command, or the official verifier ran after Agent termination. No
task met that criterion.

## Per-task result

| Task | Verifier | Intent | Source change | Agent test | Terminal / reason | Failure stage | Tokens |
|---|---|---:|---:|---:|---|---|---:|
| `abs-stepped-slices` | FAIL | no | no | no | failed / invalid intent | intent | 108,909 |
| `cliffy-config-file-parsing` | FAIL | yes | yes | no | budget exceeded / budget guard | edit | 219,970 |
| `csstree-shorthand-expansion-compression` | FAIL | no | no | no | failed / invalid intent | intent | 44,003 |
| `gql-incremental-graphql-delivery` | FAIL | no | no | no | failed / invalid intent | intent | 104,586 |
| `happy-dom-abort-pending-body-reads` | FAIL | no | no | no | failed / invalid intent | intent | 105,471 |
| `httpx-streaming-json-iteration` | FAIL | no | no | no | failed / invalid intent | intent | 91,358 |
| `katex-multicolumn-array-spans` | FAIL | no | no | no | failed / invalid intent | intent | 61,719 |
| `kombu-single-active-consumer-priority` | FAIL | no | no | no | failed / invalid intent | intent | 78,663 |
| `kombu-virtual-queue-dead-lettering` | FAIL | no | no | no | failed / invalid intent | intent | 81,394 |
| `langchain-request-coalescing` | FAIL | no | no | no | failed / invalid intent | intent | 78,219 |
| `mashumaro-flattened-dataclass-fields` | FAIL | no | no | no | failed / invalid intent | intent | 29,135 |
| `meriyah-explicit-resource-declarations` | FAIL | no | no | no | failed / invalid intent | intent | 19,184 |
| `onedump-dump-encryption-pipeline` | FAIL | no | no | no | failed / invalid intent | intent | 47,228 |
| `oxvg-structural-selector-preservation` | FAIL | no | no | no | failed / invalid intent | intent | 149,795 |
| `scriggo-method-declarations` | FAIL | no | yes | no | budget exceeded / budget guard | edit | 221,005 |
| `sqlfmt-create-table-ddl-formatting` | FAIL | no | no | no | failed / invalid intent | intent | 20,318 |
| `sqlite-utils-safe-import-checkpoints` | FAIL | no | no | no | failed / invalid intent | intent | 37,769 |
| `superjson-error-stack-serialization` | FAIL | no | no | no | failed / invalid intent | intent | 62,503 |
| `textual-kitty-key-phases` | FAIL | no | no | no | failed / invalid intent | intent | 56,263 |
| `valibot-recursive-schema-composition` | FAIL | no | no | no | failed / invalid intent | intent | 48,709 |

## Failure-stage concentration

The terminal failure attribution is:

| Stage | Tasks | Share |
|---|---:|---:|
| explore | 0 | 0% |
| readiness | 0 | 0% |
| intent | 18 | 90% |
| edit | 2 | 10% |
| test | 0 | 0% |
| finalize | 0 | 0% |

The dominant failure is therefore the **intent handoff**, not repository
exploration or source-read readiness. Nineteen tasks recorded real source-read
readiness evidence; `scriggo` instead made a source edit while still effectively
in explore and moved directly to verify. Eighteen tasks reached the handoff but
never produced an accepted structured intent and terminated after the bounded
invalid-intent path.

Only `cliffy` produced an accepted intent. It and `scriggo` made source changes,
but neither reached an actual test command or explicit finalize action before
the token budget was exceeded. Consequently this run provides no evidence about
test/finalize reliability and no successful DeepSWE solve signal.
