# Qwen3.5-4B SFT v1 holdout validation

Date: 2026-08-09

## Conclusion

**The Dataset → Train → Redeploy loop is complete, but generalized improvement has not yet been verified.**

On six unseen executable tasks, Base solved 3/6 and explicitly called `finish` on 2/6. SFT solved 4/6 and explicitly called `finish` on 3/6, but one SFT attempt was invalidated by a LocalRuntime infrastructure hang and another regressed from Base PASS + `finish` to verifier FAIL + `repeated_tool_call`. The observed +1 solve and +1 correct `finish` are encouraging, but the sample is too small and mixed—and the paired set is incomplete—to claim a clear holdout self-improvement signal.

The stricter project conclusion therefore remains:

> 闭环已完成，但泛化 improvement 尚未验证。

No training, tuning, prompt changes, AgentLoop changes, Dataset classification changes, Verifier changes, benchmark additions, or retries were performed in this stage.

## Holdout selection and training-set audit

The training checkpoint freezes a source export of 78 candidates and selects exactly 48 sample IDs. The audit used the checkpoint's `dataset_provenance.json`, not the mutable/current candidate count:

- recorded source SHA-256: `1778099b89aba57810382df462351b6efed0d688067f00c8d9937f48af074612`;
- recomputed source SHA-256: identical;
- selected IDs found: 48/48;
- unique selected task IDs / repos / base SHAs / trajectories: 22 / 22 / 22 / 48;
- duplicate selected sample IDs or source trajectory IDs: 0.

The holdout rules were:

1. existing `python-smoke-v1` tasks only; no new benchmark or task variants;
2. expected outcome `completed`, executable in LocalRuntime, and backed by a deterministic verifier;
3. task ID, repository identity, and base SHA absent from all 48 selected training samples;
4. no variant of a selected training trajectory;
5. one attempt per policy/task pair, with no retry to improve results.

The six selected tasks all passed the three identity checks:

| Holdout task | Repo | Base SHA | Training task/repo/SHA overlap |
|---|---|---|---|
| `duration-parser` | `forgeloop-eval://python-smoke-v1/duration-parser` | `1a926f88939a7dde067ec17717b264a2c158938e` | none / none / none |
| `query-encoding` | `forgeloop-eval://python-smoke-v1/query-encoding` | `89ff9f96c4fa7b75fa10729086c26ce1d9c2244f` | none / none / none |
| `atomic-reservation` | `forgeloop-eval://python-smoke-v1/atomic-reservation` | `ff0364501abc3743068568b912de86dfdb349a62` | none / none / none |
| `dependency-order` | `forgeloop-eval://python-smoke-v1/dependency-order` | `5d44195c2058f1f3527dfb33bd51e0a702f0c959` | none / none / none |
| `deep-merge` | `forgeloop-eval://python-smoke-v1/deep-merge` | `4914adaac04e173d90a85efddf6fcf971226029c` | none / none / none |
| `interval-merge` | `forgeloop-eval://python-smoke-v1/interval-merge` | `c61cca43bbc1854fd5680f1d809c777fe7d991e9` | none / none / none |

`retry-boundary` was not used because it had already appeared in the earlier Base/SFT evaluation even though it was not selected for training. `missing-vendor-contract` was not used because it is an expected-blocked environment task.

### Frozen 48-sample provenance

The table groups all selected source trajectories by their common task/repo/base identity. Counts sum to 48.

| Task ID | Repo | Base SHA | Count | Source trajectories |
|---|---|---|---:|---|
| `condition-boundary` | `forgeloop-eval://python-smoke-v1/condition-boundary` | `545aaa80dfeb89ec07efef1399ab5193e108cc45` | 7 | `condition-boundary-87308e40`, `condition-boundary-86dfaa9f`, `condition-boundary-attempt-2-2ef4ab4f`, `condition-boundary-e27b2de1`, `condition-boundary-attempt-1-33f9a02b`, `condition-boundary-4c76aba3`, `condition-boundary-71ff106e` |
| `cross-file-config` | `forgeloop-eval://python-smoke-v1/cross-file-config` | `9b97b7c39965a0a8d70ae79ec0c95d86eddc4e99` | 2 | `cross-file-config-attempt-2-054e9773`, `cross-file-config-attempt-1-62565447` |
| `dataclass-serialization` | `forgeloop-eval://python-smoke-v1/dataclass-serialization` | `1a04a7b779c948f14a82842b26244f09a280b998` | 2 | `dataclass-serialization-attempt-2-d56296fd`, `dataclass-serialization-attempt-1-4a706169` |
| `datetime-api` | `forgeloop-eval://python-smoke-v1/datetime-api` | `37ffb8e59205fcc0180a4e1c54acc76e2d8d74c4` | 1 | `datetime-api-attempt-2-aafe2d08` |
| `dict-missing` | `forgeloop-eval://python-smoke-v1/dict-missing` | `62aeb47b4396874d749aef55c84b2c7430f1656b` | 2 | `dict-missing-attempt-2-384c452f`, `dict-missing-attempt-1-3461b8f3` |
| `email-regression` | `forgeloop-eval://python-smoke-v1/email-regression` | `9fd6b758274130b277d3916c8e5793edd8b0ac1e` | 2 | `email-regression-attempt-1-df5046d8`, `email-regression-attempt-2-d680ac4f` |
| `mutable-default` | `forgeloop-eval://python-smoke-v1/mutable-default` | `839b65630d049bafebdd893526fc2760354dcd86` | 2 | `mutable-default-attempt-1-b8d9b70c`, `mutable-default-attempt-2-beb6c957` |
| `none-handling` | `forgeloop-eval://python-smoke-v1/none-handling` | `59b83967e38ba4ae363e69816b46bf7391aaab27` | 4 | `none-handling-2e3b518a`, `none-handling-attempt-2-a92d6bf1`, `none-handling-1ac5e3cd`, `none-handling-attempt-1-fb9c7c35` |
| `off-by-one` | `forgeloop-eval://python-smoke-v1/off-by-one` | `e198d36859eff31e17bfada0a49bf2f8460f1804` | 2 | `off-by-one-attempt-1-2d981fe7`, `off-by-one-attempt-2-6a83de04` |
| `parameter-validation` | `forgeloop-eval://python-smoke-v1/parameter-validation` | `3af423c3bdabb2b1feac5c76d56cd22500520d31` | 3 | `parameter-validation-d6380c4b`, `parameter-validation-attempt-2-476d9c97`, `parameter-validation-attempt-1-49e27749` |
| `path-suffix` | `forgeloop-eval://python-smoke-v1/path-suffix` | `5b5fc2b97310539b045d41bac0f4f6ed2669adea` | 2 | `path-suffix-attempt-2-fb54600f`, `path-suffix-attempt-1-54a874ca` |
| `recursive-flatten` | `forgeloop-eval://python-smoke-v1/recursive-flatten` | `81c4ca31b926d556033cc46f1bdfe7b3a503f199` | 2 | `recursive-flatten-attempt-1-54092dc3`, `recursive-flatten-attempt-2-3cf000b4` |
| `regression-test` | `forgeloop-eval://python-smoke-v1/regression-test` | `6c24101af77e58ee0b39084dce060b8fc97d07d4` | 1 | `regression-test-attempt-1-89b73b8e` |
| `root-cause-cache` | `forgeloop-eval://python-smoke-v1/root-cause-cache` | `9bd711988f7eb82985e041644c4c538016a57e57` | 2 | `root-cause-cache-attempt-2-95479658`, `root-cause-cache-attempt-1-cd1073dd` |
| `safe-ratio` | `forgeloop-eval://python-smoke-v1/safe-ratio` | `7cd4f99fa8d81e95d5cb8558a58a95ce7dc69dad` | 2 | `safe-ratio-attempt-1-c3090c35`, `safe-ratio-attempt-2-ef1fadf8` |
| `small-refactor` | `forgeloop-eval://python-smoke-v1/small-refactor` | `54428a5f1ccfaa834aee6e033421459864bda82d` | 1 | `small-refactor-attempt-1-5d11c01c` |
| `stable-sort` | `forgeloop-eval://python-smoke-v1/stable-sort` | `d97c932b51665f72feaf7a910a438d80f925d5e3` | 2 | `stable-sort-attempt-2-7c2ecc86`, `stable-sort-attempt-1-df9b7d85` |
| `status-mapping` | `forgeloop-eval://python-smoke-v1/status-mapping` | `7fcfcfb02d133e381337d694450a6b10e358d60b` | 2 | `status-mapping-attempt-2-eab2b6f1`, `status-mapping-attempt-1-0d1919c2` |
| `string-normalization` | `forgeloop-eval://python-smoke-v1/string-normalization` | `dcdbfeb922120763d7ba6bf9a7e4ae08e77ef065` | 2 | `string-normalization-attempt-2-b305c82c`, `string-normalization-attempt-1-d4d03f4f` |
| `timezone-deadline` | `forgeloop-eval://python-smoke-v1/timezone-deadline` | `4c661de112a934835abb15c13403ea9f73223a95` | 1 | `timezone-deadline-attempt-1-ff88edce` |
| `ttl-cache` | `forgeloop-eval://python-smoke-v1/ttl-cache` | `840f9da155538d9315e288175cb607ca3b8cdf77` | 1 | `ttl-cache-attempt-1-99499700` |
| `two-file-api-rename` | `forgeloop-eval://python-smoke-v1/two-file-api-rename` | `90fc572afaaf4db1473c0035a1b6cd41ba790459` | 3 | `two-file-api-rename-303d60ec`, `two-file-api-rename-attempt-1-b4097534`, `two-file-api-rename-attempt-2-5f8e7fed` |

## Controlled execution

Both policies used Ollama through `http://127.0.0.1:11434/v1`, LocalRuntime, an 8192-token context, 2048 maximum output tokens, temperature 0.2, top-p 0.95, and non-streaming generation. Tasks, task modes, prompts, budgets (`30` steps, `30` model calls, `80` tool calls, `500000` tokens, task-defined wall timeout), and verifiers were identical. The configured experimental variable was only policy/model identity.

Every Base task ran exactly once. Every SFT task also ran exactly once. After the fourth SFT attempt hung in LocalRuntime cleanup, that attempt was preserved as infrastructure failure; only the two tasks that had not started were subsequently executed. The timeout-cleanup regression fix was present for those final two SFT attempts, which is a harness-version difference; neither attempt exercised the timeout path, and AgentLoop/runtime behavior outside timeout cleanup was unchanged. No model attempt was retried.

## Per-task results

Tokens for SFT `duration-parser` are reconstructed by summing its 11 successful `model_response` usage records because the final provider request failed and the task-level aggregate is therefore null. The interrupted `dependency-order` row likewise uses its five recorded responses. `>780s` is a conservative lower bound for the infrastructure hang, not valid model latency.

| Task | Policy | Verifier | Explicit `finish` | Terminal / stop reason | Steps / model / tool | Input / output tokens | Wall time | Dataset classification | Repeated tool error / no_progress / malformed |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| `duration-parser` | Base | FAIL | no | failed / `model_final_message` | 9 / 9 / 8 | 35,797 / 4,318 | 87.723s | `model_failure` | no / no / no |
|  | SFT | PASS | no | failed / `orchestration_error` | 12 / 12 / 11 | 48,955 / 4,033 | 99.844s | `successful_but_inefficient` | no / no / no |
| `query-encoding` | Base | PASS | yes | completed / `model_finish_tool` | 7 / 7 / 7 | 12,681 / 1,083 | 23.174s | `sft_candidate` | no / no / no |
|  | SFT | FAIL | no | blocked / `repeated_tool_call` | 5 / 5 / 5 | 8,038 / 352 | 9.709s | `model_failure` | no / no / no; repeated identical call: yes |
| `atomic-reservation` | Base | PASS | no | completed / `model_final_message` | 8 / 8 / 7 | 16,529 / 1,376 | 28.507s | `sft_candidate` | no / no / no |
|  | SFT | PASS | yes | completed / `model_finish_tool` | 7 / 7 / 8 | 13,486 / 934 | 21.392s | `sft_candidate` | no / no / no |
| `dependency-order` | Base | FAIL | no | failed / `model_final_message` | 16 / 16 / 15 | 62,092 / 5,347 | 104.154s | `model_failure` | no / no / no |
|  | SFT | not reached | no | infrastructure failure / interrupted partial trajectory | 5 / 5 / 5 | 9,240 / 833 | >780s | `infrastructure_failure` (not indexed; no task result) | no / no / no |
| `deep-merge` | Base | FAIL | no | failed / `model_final_message` | 13 / 13 / 12 | 48,020 / 3,682 | 79.645s | `model_failure` | no / no / no |
|  | SFT | PASS | yes | completed / `model_finish_tool` | 12 / 12 / 12 | 39,025 / 3,050 | 65.177s | `successful_but_inefficient` | no / no / no |
| `interval-merge` | Base | PASS | yes | completed / `model_finish_tool` | 8 / 8 / 8 | 16,747 / 1,312 | 28.575s | `sft_candidate` | no / no / no |
|  | SFT | PASS | yes | completed / `model_finish_tool` | 10 / 10 / 10 | 23,570 / 1,829 | 36.956s | `sft_candidate` | no / no / no |

`repeated_tool_call` is reported separately from repeated tool **errors**: the SFT `query-encoding` reads were valid and successful but repeated without repository progress. There were no malformed tool-call payloads and no `no_progress` terminal guard in any trajectory.

## Aggregate comparison

| Metric | Base | SFT |
|---|---:|---:|
| Verifier PASS across planned six | 3/6 (50.0%) | 4/6 (66.7%); one verifier not reached |
| Explicit correct `finish` | 2/6 (33.3%) | 3/6 (50.0%) |
| Completed terminal state | 3/6 | 3/6 |
| Model failure | 3/6 | 1/6 |
| Infrastructure failure | 0/6 | 1/6 |
| Recorded input tokens | 191,866 | 142,314 |
| Recorded output tokens | 17,118 | 11,031 |
| Local API cost | $0 | $0 |
| Dataset: `sft_candidate` | 3 | 2 |
| Dataset: `successful_but_inefficient` | 0 | 2 |
| Dataset: `model_failure` | 3 | 1 |
| Dataset: `infrastructure_failure` | 0 | 1 (manual attribution; partial run was not indexable) |

The five verifier-complete SFT attempts solved 4/5 versus Base's 3/5 on the same five tasks. This is a positive observation, not a generalization claim: SFT improved `duration-parser` and `deep-merge`, preserved `atomic-reservation` and `interval-merge`, and regressed `query-encoding`.

## Harness issue found and fixed

The SFT `dependency-order` test command timed out under Windows. `subprocess.run(timeout=...)` killed its PowerShell parent, but an orphaned `pytest` descendant retained the captured stdout/stderr pipe. Python then waited indefinitely for pipe EOF, bypassing the task wall-clock limit.

LocalRuntime now launches the command explicitly and, on Windows timeout, terminates the exact process tree before collecting output. A Windows-only regression test spawns a child that retains the pipe and verifies that the runtime returns a timed-out result promptly and leaves no child process. This changes only Runtime cleanup; AgentLoop, prompts, Dataset rules, and Verifier behavior are unchanged.

All 12 recorded trajectories—including the partial infrastructure trajectory—successfully passed offline `trace replay` and `trace explain`. Dataset indexes were built from the completed Base and SFT task records; the partial infrastructure attempt was intentionally not converted into a fabricated task result.
