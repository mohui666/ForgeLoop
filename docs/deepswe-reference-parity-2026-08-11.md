# DeepSWE Reference Parity Test

Follow-up: [Execution Budget v2](execution-budget-v2-2026-08-11.md) removes
cumulative provider token usage from the execution horizon and records the
resulting one-shot validation.

Date: 2026-08-11

ForgeLoop commit under test: `c101a60762d8520e246bdba0bce770fb21d7e9d4`

DeepSWE checkout: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`

## Executive conclusion

The comparison confirms a real execution-horizon difference, not a general inability of DeepSeek V4-Flash to edit these repositories.

- In the released `mini-swe-agent` harness, V4-Flash produced real source edits on all four tasks and entered validation. Its first source edit occurred after 0.62M to 2.22M cumulative input tokens.
- In the three valid ForgeLoop runs, V4-Flash produced no source edit. ForgeLoop stopped at 150k to 166k cumulative input-plus-output tokens with `controller_exploration_exhausted`.
- ForgeLoop counts every response's full input/context tokens again in the 200k execution budget. The controller then ends pre-edit exploration at 75% of that budget. This makes repeated long-horizon context presentation the dominant execution limit even when most reference-harness input is cached.
- The strongest supported ForgeLoop difference is therefore budget semantics and the resulting execution horizon. Tool protocol, prompt, and context presentation also differ, but this test did not isolate them with an ablation.
- Final patch/verifier parity was not valid on the reference side: the local released Pier 0.3.0 did not run the DeepSWE v1.1 `[[verifier.collect]]` hook. Three mini-swe-agent runs committed real changes, but Pier never created `/logs/artifacts/model.patch`, so the verifier graded pristine source.
- Mashumaro exhausted the DeepSeek account balance late in the A run. The A trajectory still proves edit/test behavior, but A did not finish and B failed its first provider request. Mashumaro is not a valid end-to-end A/B comparison.

No ForgeLoop controller, prompt, budget, context, model, task, verifier, or collector behavior was changed during this test.

## Harness and request configuration

### A: DeepSWE reference command

The command shape follows DeepSWE's documented single-task Pier invocation and uses mini-swe-agent 2.4.6:

```powershell
uv run pier run --path .forgeloop/external/deep-swe/tasks/<task> `
  --agent mini-swe-agent `
  --model deepseek/deepseek-v4-flash `
  --agent-kwarg reasoning_effort=max `
  --agent-kwarg 'model_kwargs={"api_base":"https://api.deepseek.com/v1","extra_body":{"thinking":{"type":"enabled"}},"max_tokens":8192}' `
  --env docker --n-attempts 1 --n-concurrent 1 --max-retries 0
```

The saved raw mini-swe-agent trajectories confirm the final request-builder configuration:

```json
{
  "drop_params": true,
  "reasoning_effort": "max",
  "api_base": "https://api.deepseek.com/v1",
  "extra_body": {"thinking": {"type": "enabled"}},
  "max_tokens": 8192
}
```

mini-swe-agent had `step_limit=0`, `cost_limit=0.0`, and `wall_time_limit_seconds=0`; the official task supplied the outer 5,400-second agent timeout.

### B: frozen ForgeLoop

The frozen policy was `deepseek-v4-flash-controller-v1.3-simplified` with Execution Closure v2, Context Efficiency v1, and Agent Policy execution-first v1. Its recorded serving/generation identity confirms:

```json
{
  "api_base": "https://api.deepseek.com/v1",
  "thinking_level": "max",
  "extra_body": {"thinking": {"type": "enabled"}},
  "max_tokens": 8192,
  "stream": false
}
```

ForgeLoop's request builder maps `max` to both `reasoning_effort="max"` and DeepSeek's explicit thinking body, then passes the merged kwargs to LiteLLM. The run limits were 30 model calls, 30 steps, 80 tool calls, 1,800 seconds, and 200,000 cumulative input-plus-output tokens.

This audit confirms `reasoning_effort=max` for both groups at the actual LiteLLM request-builder boundary and in persisted run configuration. It did not use an HTTP MITM capture.

## Execution comparison

Call positions below are successful assistant action indexes. Reference `API calls` include provider attempts/format retries, so they can be slightly larger than agent steps. Token counts at first test/edit are cumulative input/output at that action.

| Task | Harness | Steps / API calls | First test or validation | First real source edit | Source diff | Finish |
|---|---|---:|---|---|---|---|
| abs-stepped-slices | A mini | 102 / 103 | call 18, reproduction, 295,036 / 2,798 | call 27, `ast/ast.go`, 623,582 / 8,831 | yes; committed as `7ac692e` | submitted |
| abs-stepped-slices | B ForgeLoop | 17 / 17 | none | none | no | no; exploration exhausted |
| sqlfmt-create-table-ddl-formatting | A mini | 130 / 133 | call 15, formatter reproduction, 384,350 / 3,256; pytest at 16 | call 38, `src/sqlfmt/ddl.py`, 2,219,617 / 29,859 | yes; committed as `ea66cae` | submitted |
| sqlfmt-create-table-ddl-formatting | B ForgeLoop | 13 / 13 | none | none | no | no; exploration exhausted |
| sqlite-utils-safe-import-checkpoints | A mini | 207 / 207 | call 15, SQLite transaction reproduction, 183,006 / 7,905 | call 39, `sqlite_utils/db.py`, 1,366,106 / 36,080 | yes; committed as `628ac65` | submitted |
| sqlite-utils-safe-import-checkpoints | B ForgeLoop | 15 / 15 | call 13, savepoint reproduction, 115,846 / 4,763 | none | no | no; exploration exhausted |
| mashumaro-flattened-dataclass-fields | A mini | 156 / 158 | call 37, flatten reproduction, 1,038,624 / 11,776 | call 45, `mashumaro/helper.py`, 1,512,921 / 28,027 | yes; uncommitted at interruption | no; insufficient balance |
| mashumaro-flattened-dataclass-fields | B ForgeLoop | 1 / 1 | none | none | no | no; insufficient balance before response |

The reference runs did not merely invoke edit-looking commands. Their trajectories contain Git-visible source diffs; ABS, SQLfmt, and SQLite subsequently committed those diffs. SQLfmt also ended with `1243 passed` before commit and submission. Mashumaro had a visible `mashumaro/helper.py` diff and continued running targeted and full tests before the provider rejected a later call.

## Usage, termination, collector, and verifier

| Task | Harness | Input / cached / output tokens | Time | API cost | Termination | `model.patch` | Official verifier |
|---|---|---:|---:|---:|---|---|---|
| ABS | A mini | 6,892,536 / 6,822,912 / 62,991 | 953.309s | $0.046489 | submitted | absent: Pier collect hook did not run | 0; F2P 0/6, P2P 6/6, partial 0.500000 |
| ABS | B ForgeLoop | 158,301 / 46,976 / 7,523 | 226.256s | $0.017823 | `controller_exploration_exhausted` | absent: no source change | 0; F2P 0/6, P2P 6/6, partial 0.500000 |
| SQLfmt | A mini | 18,101,299 / 17,984,896 / 129,768 | 1,599.109s | $0.102989 | submitted | absent: Pier collect hook did not run | 0; F2P 0/32, P2P 1273/1273, partial 0.975479 |
| SQLfmt | B ForgeLoop | 150,957 / 32,640 / 3,199 | 190.593s | $0.017551 | `controller_exploration_exhausted` | absent: no source change | 0; F2P 0/32, P2P 1273/1273, partial 0.975479 |
| SQLite | A mini | 23,323,060 / 23,225,088 / 113,761 | 1,331.977s | $0.110599 | submitted | absent: Pier collect hook did not run | 0; F2P 0/60, P2P 1038/1038, partial 0.945355 |
| SQLite | B ForgeLoop | 149,400 / 30,464 / 8,995 | 196.500s | $0.019255 | `controller_exploration_exhausted` | absent: no source change | 0; F2P 0/60, P2P 1038/1038, partial 0.945355 |
| Mashumaro | A mini | 16,196,887 / 16,109,056 / 103,928 | 1,576.583s | $0.086502 | provider `Insufficient Balance` | absent: interrupted before commit/collect | 0; F2P 0/66, P2P 30014/30014, partial 0.997806 |
| Mashumaro | B ForgeLoop | unavailable | 127.716s | $0 | first request: `Insufficient Balance` | absent | 0; F2P 0/66, P2P 30014/30014, partial 0.997806 |

Measured model API cost was $0.346579 for A and $0.054630 for B, $0.401209 total. Local qwen advisory classification has no API cost.

The `pier run` process for completed A trials returned a post-result Windows console error because Rich attempted to print a bullet character through the legacy GBK encoder. Trial result files nevertheless record one completed, non-errored trial and the agent exit status `Submitted`. This display failure is separate from agent termination and verification.

## Reference collector qualification failure

DeepSWE's checked-out README says v1.1 separate-verifier grading requires Pier newer than 0.3.0. ForgeLoop's `deepswe` extra and lockfile pin exactly `datacurve-pier==0.3.0`, and PyPI exposed no version newer than 0.3.0 during the test. Even the current Pier Git revision `34520b5f00ecec106f622fcbbb4f88ecf8401ff0` still reports version 0.3.0, although its implementation may be ahead of the released package.

All A runs used the released pinned 0.3.0. Their job logs show Pier trying to download `/logs/artifacts/model.patch` and receiving `Could not find the file`; there is no preceding collect-hook execution. Therefore:

1. The A agent behavior evidence—calls, edits, tests, commits, tokens, and submission—is valid.
2. The A patch-delivery and verifier result is not a valid test of the model's committed patch.
3. The pristine verifier scores must not be interpreted as patch quality or solve rate.

This upstream/version incompatibility was discovered during the diagnostic and was not repaired, in accordance with the no-modification scope.

## Budget semantics audit

ForgeLoop's `BudgetState.record_usage` adds every provider response's full `input_tokens` and `output_tokens` to cumulative counters. `total_tokens` is their sum. Cached tokens are recorded only as a parallel metric and are not subtracted. The DeepSWE adapter sets `max_tokens=200000` by default.

The frozen controller reads this same cumulative `total_tokens` and terminates the explore phase at `0.75 * max_tokens`, nominally 150,000 tokens. The observed stops match that implementation:

- ABS: 165,824 total after the last accepted response.
- SQLfmt: 154,156 total.
- SQLite: 158,395 total.

The overshoot is expected because usage is checked between complete model calls. The response that crosses 150k is retained, then the controller terminates.

This accounting is valid as aggregate provider-usage telemetry, but it is not parity-equivalent to the reference harness's long-horizon execution budget. The reference harness had no token/cost/step hard limit and reused a heavily cached growing context: 98.99% to 99.58% of its input tokens were reported cached. Re-counting the full prompt on every round as the principal execution horizon makes context replay, rather than new reasoning or actions, consume ForgeLoop's pre-edit allowance.

Most importantly, every reference first edit occurred after the entire ForgeLoop run would already have terminated:

- ABS: 623,582 input tokens, 3.8 times ForgeLoop's final total.
- SQLite: 1,366,106 input tokens, 8.6 times ForgeLoop's final total.
- Mashumaro: 1,512,921 input tokens, 9.1 times the valid ForgeLoop stop range.
- SQLfmt: 2,219,617 input tokens, 14.4 times ForgeLoop's final total.

This confirms that the current budget semantics are sufficient to produce the observed ForgeLoop 0-edit behavior relative to mini-swe-agent. It does not prove that a larger horizon alone would produce correct patches.

## Other harness differences

These remain plausible secondary contributors but were not isolated:

- mini-swe-agent presents one mandatory Bash tool and requires one command per response; ForgeLoop presents multiple structured tools and allows multiple tool calls.
- mini-swe-agent retains a straightforward conversation and truncates only observations over 10,000 characters; ForgeLoop applies deterministic context compaction and provenance presentation.
- The system prompts differ. mini-swe-agent explicitly prescribes repository analysis, reproduction, edit, verification, edge tests, and a submit command. ForgeLoop uses its frozen execution-first policy.
- ForgeLoop has advisory qwen classification and deterministic execution closure. The reference harness has no controller.
- ForgeLoop has deterministic Git patch delivery after a real edit. The reference harness relies on the model committing and on Pier's collect hook.

The timing evidence makes budget/execution horizon the highest-confidence difference: the same model's first edits are all beyond ForgeLoop's stop point. The current experiment cannot rank the remaining differences without changing one variable at a time.

## Single highest-priority next change

Redefine ForgeLoop's execution budget so cumulative input and cached-context tokens remain accounting telemetry but are not the pre-edit hard-stop horizon. Use one execution-oriented allowance—model calls/wall time or uncached/new-context tokens—for the long-horizon stop decision. This is one budget-semantics change; prompt, tool protocol, context policy, and controller behavior should remain frozen for its first validation.

Before rerunning final delivery parity, the diagnostic harness also needs a Pier build that actually supports DeepSWE v1.1 collect hooks and a replenished DeepSeek balance. Those are test prerequisites, not additional ForgeLoop behavior recommendations.
