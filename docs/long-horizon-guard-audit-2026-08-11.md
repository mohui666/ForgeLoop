# Long-Horizon Guard Audit & Fix

Date: 2026-08-11

Implementation base: `07408dea30287492a38e02b21427206ec14f9880`

Policy under test: `deepseek-v4-flash-controller-v1.3-simplified`

Execution horizon: 256 model calls / 256 steps / 1,024 tool calls / 5,400s

## Result

The audit removed short-task terminal assumptions from the active DeepSWE
execution path. Cumulative input, cached, reasoning, and output usage remains
accounting telemetry. Controller phases no longer own call or token deadlines.
The only action-repetition hard stop now requires a contiguous sequence with
the same canonical tool name and arguments, the same visible `ok + output`
observation, and no Git workspace fingerprint change.

The live result is 0/3 solved, but the guard objective is demonstrated. ABS
continued from the old 47-call termination to 154 model calls, first edited at
call 78, first validated at call 115, ran 12 successful internal validations,
and explicitly finished. It had no repeated-action guard event. SQLfmt and
SQLite also had no repeated-action event; both ended on genuine provider
connection failures rather than a deterministic ForgeLoop guard.

## Repeated-action semantics

- Action fingerprint: SHA-256 of canonical JSON containing exact tool name and
  arguments. Different paths, ranges, commands, or arguments are different
  actions.
- Observation fingerprint: SHA-256 of exact tool `ok` and `output`. New output
  resets the streak.
- Progress state: Git progress fingerprint before and after the action. Any
  workspace change resets the streak, as does any intervening different action.
- Window: contiguous tool actions only; there is no run-global counter.
- Default warning: streak 2.
- Configured repeat limit: 3; hard stop occurs only after the fourth action has
  actually executed and returned the same observation with no workspace
  progress.
- Provenance schema: `forgeloop.long-horizon-guards.v1`. Resolved thresholds
  are present in `provenance.json` and the trajectory `run_started` event.

`repeated_error` and generic mutation `no_progress` are no longer terminal
reasons. Repeated errors remain explicit tool observations and recovery
feedback. Consecutive unchanged `apply_patch` attempts produce an advisory.

The old ABS repeats were calls 28/32/42/47 and the old SQLfmt repeats were
calls 31/35/51/57. They were separated by other actions and evidence, so the
old global count was not proof of a loop. The new runs recorded zero repeat
warnings and zero repeat terminals.

## Post-edit closure semantics

The following V1.3 terminal deadlines were removed:

- validation within 4 calls or 60,000 phase tokens;
- repair within 5 calls or 70,000 phase tokens;
- diff review within 2 calls or 40,000 phase tokens;
- post-pass finalization within 3 calls or 40,000 tokens;
- terminal after 6 validation attempts;
- exploration terminal at 75% of a legacy `max_tokens` snapshot;
- terminal after two premature `finish(completed)` requests.

Validation, repair, and diff-review reminders are now one-time advisories at 8,
8, and 4 phase calls respectively. They do not reduce the execution horizon.
Premature finish/plain-final loops terminate only after four consecutive,
identical requests with the same workspace and phase state. A stable validated
and reviewed tree still permits explicit finish and retains the existing
`controller_ready_auto_finish` completion path.

The SQLite regression is covered directly: an edit at call 98 followed by two
large edit responses and more than the old 60,000 phase-token allocation does
not terminate before call 101. The test continues through call 200 and can
validate at call 201. A separate repair test passes after seven validation
attempts and multi-million cumulative tokens.

## Terminal audit

| Terminal surface | Active DeepSWE | Decision |
|---|---:|---|
| model-call / step limit | yes | keep; 256 resolved calls |
| tool-call limit | yes | keep; 1,024 calls |
| wall-clock timeout | yes | keep; 5,400s |
| optional cost safety | configurable | keep; independent of token replay |
| provider timeout/failure | yes | keep; external failure is recorded |
| orchestration exception | yes | keep; boundary failure is recorded |
| user interrupt | yes | keep |
| contiguous identical no-change tool loop | yes | keep with strict v1 evidence |
| run-global repeated tool/error counters | no | removed as terminals |
| shell/apply-patch no-Git-progress counter | no | removed as terminal |
| V1.3 exploration/validation/repair/review/finalization deadlines | no | removed as terminals |
| repeated invalid completion request | yes | strict identical four-request loop only |
| validated + reviewed stable-tree auto-finish | yes | keep; successful completion |
| model `finish` / plain final | yes | keep; model terminal decision |
| delivery failure after claimed completion | yes | keep; cannot report undeliverable completion |
| legacy edit-intent handoff failure | not used by V1.3 Simplified | unchanged; opt-in legacy controllers can still emit `controller_invalid_edit_intent` |

No remaining active DeepSWE terminal depends on cumulative provider tokens or
a fixed post-edit phase allocation.

## Canary

The unchanged `query-encoding` canary passed once with the refreshed Windows
User credential.

| Result | Calls / tools | Input / cached / output | Wall time | Cost | Termination |
|---|---:|---:|---:|---:|---|
| PASS | 6 / 8 | 17,371 / 14,720 / 1,169 | 36.592s | $0.00073968 | `model_finish_tool` |

First edit was call 3 and first validation was call 4. No repeated-action event
was recorded. Artifact:
`.forgeloop/long-horizon-guard-audit-canary-valid-20260811/20260811T143545Z-f034600e`.

One preceding infrastructure preflight inherited the stale process credential
and failed its first request with `Insufficient Balance`; it made zero tool
calls and has unavailable usage. It is retained but is not a model-behavior
attempt.

## DeepSWE one-shot results

Calls below are successful model responses except the terminal provider request,
which is included in the reported model-call count.

| Task | Official result | Calls / tools | First behavioral test | First source edit | Edit -> validation | Repeat guard | Termination |
|---|---|---:|---|---|---:|---:|---|
| ABS | FAIL | 154 / 201 | call 115, PASS; 3,076,612 / 1,888,384 / 154,058 | call 78, `ast/ast.go`; 1,721,292 / 884,352 / 114,068 | 37 calls | 0 warning / 0 terminal | `completed/model_finish_tool` |
| SQLfmt | FAIL | 17 / 23 | formatter probe call 8, PASS; 48,976 / 20,736 / 1,561 | none | n/a | 0 / 0 | `failed/provider_failure` (SSL EOF) |
| SQLite | FAIL | 36 / 54 | none | none | n/a | 0 / 0 | `failed/provider_failure` (incomplete chunked response) |

Usage in action columns is cumulative input / cached / output at that call.

| Task | Final or recorded input / cached / output | Reasoning | Wall time | Cost | Edit / test / finish | Patch state |
|---|---:|---:|---:|---:|---|---|
| ABS | 5,359,953 / 3,359,488 / 184,206 | 148,961 | 2,437.977s | $0.34104935 | yes / yes / yes | ForgeLoop delivery committed 27,843 bytes across five source/test files; Pier 0.3.0 collected no `model.patch` |
| SQLfmt | >=151,783 / >=46,208 / >=4,583 | >=2,860 | 238.573s | >=$0.01619312 | no / probe only / no | no changes |
| SQLite | >=536,714 / >=163,328 / >=18,268 | >=13,873 | 436.523s | >=$0.05784640 | no / no / no | no changes |

For SQLfmt and SQLite, 16 and 35 responses respectively returned complete
provider usage. The following request failed before returning usage, so the
run total is correctly unavailable; the table reports only the auditable lower
bound and does not treat the failed request as zero.

ABS changed `ast/ast.go`, `parser/parser.go`, `evaluator/evaluator.go`,
`parser/parser_test.go`, and `evaluator/evaluator_test.go`. All 12 recorded
internal validations passed. Its first premature finish at call 152 was
recovered; after diff review the model finished at call 154. The official
verifier still received no collected patch, so its F2P result remained 0/6.
The known Pier collector/version issue was deliberately not changed.

Artifacts:

- `.forgeloop/eval-v2-runs/long-horizon-guard-audit-abs-20260811`
- `.forgeloop/eval-v2-runs/long-horizon-guard-audit-sqlfmt-20260811`
- `.forgeloop/eval-v2-runs/long-horizon-guard-audit-sqlite-20260811`

## Regression and packaging evidence

New tests cover contiguous same-action/same-observation/no-workspace-change
proof, reset on new observation/different scope/workspace progress,
non-contiguous repeated reads, warning-before-terminal behavior, removal of
`repeated_error` and `no_progress` terminals, SQLite's call-98/token regression,
repair beyond six failed validations, and provenance serialization.

Final gates:

```powershell
uv run pytest
uv run ruff check .
uv build
git diff --check
```

The prompt, V4-Flash policy, Context Efficiency, advisory 1.5B classifier,
DeepSWE tasks/verifiers/collector, and training scope were unchanged.
