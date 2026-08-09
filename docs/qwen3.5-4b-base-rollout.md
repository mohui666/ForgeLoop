# Qwen3.5-4B Base rollout

Date: 2026-08-09

This is a four-task, one-attempt-per-task stability check for the active
`qwen3.5-4b-local` Base Policy. It is not a benchmark run and did not change the
tasks, prompt, `AgentLoop`, Dataset classification, or Verifier. Ollama served
the pinned Q4_K_M artifact through `http://127.0.0.1:11434/v1` with an 8192-token
context and 2048-token output limit. Docker was unavailable, so all tasks used
`LocalRuntime`.

The source run is
`.forgeloop/eval-runs/qwen35-4b-base-rollout/20260809T033045Z-67d7a9c9`.
These local evidence artifacts are intentionally excluded from Git.

## Results

| Task type | Task | Verifier | Terminal state | Steps / model / tool calls | Input / output tokens | Wall time | Dataset classification | Error signals | Attribution |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Test-failure diagnosis | `root-cause-cache` | PASS | `completed/model_finish_tool` | 9 / 9 / 9 | 18,334 / 1,408 | 37.287s | `sft_candidate` | no repeated tool error, no no-progress, no malformed call | none |
| Simple bug fix | `datetime-api` | PASS | `failed/model_final_message` | 9 / 9 / 8 | 17,565 / 1,296 | 33.837s | `successful_but_inefficient` (`terminal_state:failed`) | no repeated tool error, no no-progress, no malformed call | Model: omitted the required `finish` call after a passing test |
| Boundary logic | `retry-boundary` | PASS | `failed/model_final_message` | 6 / 6 / 5 | 11,832 / 1,192 | 28.711s | `successful_but_inefficient` (`terminal_state:failed`) | no repeated tool error, no no-progress, no malformed call | Model: omitted the required `finish` call after making the verified fix |
| Cross-file change | `cross-file-config` | PASS | `failed/model_final_message` | 12 / 12 / 12 | 23,400 / 1,156 | 33.671s | `successful_but_inefficient` (`terminal_state:failed`) | no repeated tool error, no no-progress, no malformed call | Model: omitted the required `finish` call after a passing test |

All four task verifiers passed on their only attempt. The batch used 71,131
input tokens and 5,052 output tokens (76,183 total), took 133.506 seconds in
aggregate, and recorded zero local API cost. It produced 38 Effect Events:
17 `file.read`, 5 `file.write`, 5 `git.change`, 6 `shell.exec`, and 5
`test.run`. Every trajectory replayed and explained offline, and none carried a
safety flag.

Some individual tool calls failed and were recovered: the model tried an
incorrect relative path, used Bash `&&` once in PowerShell, and observed expected
pre-fix test failures. No task repeated the same failed tool observation enough
to trigger `repeated_tool_error`; none ended through `no_progress`; all tool-call
arguments reached the typed tool boundary without a malformed-call failure.

## Dataset and SFT gate

The isolated four-task Dataset contains:

- 1 `sft_candidate`;
- 3 `successful_but_inefficient`;
- 0 `model_failure`;
- 0 `infrastructure_failure`.

Across all six recorded trajectories for policy `qwen3.5-4b-local`, including
the earlier integration gate, the current counts are:

- 1 `sft_candidate`;
- 4 `successful_but_inefficient`;
- 0 `model_failure`;
- 1 historical `infrastructure_failure` from the pre-fix local proxy failure.

The strict SFT adapter exports only `root-cause-cache`. The 4/4 executable task
success shows that model inference, tool calling, LocalRuntime, Verifier,
trajectory, Effects, replay/explain, and Dataset paths are stable. It does not
yet establish Base Policy stability for a first SFT: three of four new runs
failed the explicit terminal protocol, and one eligible conversation is too
little training data for a meaningful first run. The truthful readiness verdict
is therefore **not ready yet**. No prompt workaround, task change, retry, SFT, or
RL run was performed.
