# Prompt Cache Audit and Pi-Parity Telemetry

Date: 2026-08-13

## Outcome

ForgeLoop's DeepSeek request prefix is already cache-stable on the current
`deepseek-v4-flash-controller-v1.3-simplified` path. The apparent low rate was
primarily a metric-definition problem: cumulative `cached/input` mixes the
unavoidable first cold request with warm requests. ForgeLoop now preserves that
provider accounting metric and separately records Pi-style warm reusable-prefix
efficiency.

Three ordinary Stage B tasks passed. Their weighted warm-prefix hit rate was
**41,659 / 42,263 = 98.5709%**, above the requested 98% average gate. No hard or
DeepSWE task was started for this audit.

## References inspected

- Pi checkout: `badlogic/pi-mono` at
  `581d75a89cea21e50d6a26df840352f94427f633`.
- Pi `packages/coding-agent/src/core/cache-stats.ts`: compare each completed
  request with the preceding prompt, use `min(previous, current)` as reusable
  input, reset after compaction/branch changes, and ignore per-turn misses at or
  below 1,024 tokens as cache-breakpoint noise.
- Pi `packages/coding-agent/src/core/compaction/compaction.ts`: compact near the
  context boundary, then rebuild a summary plus retained recent messages.
- DeepSeek official context-caching guide:
  <https://api-docs.deepseek.com/guides/kv_cache>. Caching is automatic and
  best-effort; it requires an identical prefix from token zero and persists
  prefix units at request/model-output boundaries and fixed intervals.
- DeepSeek official Chat Completion schema:
  <https://api-docs.deepseek.com/api/create-chat-completion>. `prompt_tokens`
  equals `prompt_cache_hit_tokens + prompt_cache_miss_tokens`, so ForgeLoop's
  original cached/input quotient remains valid raw provider accounting.

## Full request-path audit

| Surface | Finding | Action |
|---|---|---|
| system/task prefix | Stable within a trajectory. The workspace path is task-specific, so a new task still has a legitimate cold segment. | No prompt change. |
| tool schemas | V1.3 Simplified delegates to `ControllerV1` and keeps the same schemas. All 18 audited live requests used identical schemas within their task. Other controller policies may intentionally change schemas. | Record a byte-order-sensitive schema and request-prefix fingerprint; treat a change as a new measurement epoch. |
| message history | All three live tasks were exact append-only extensions on every warm request. | Preserve `AgentMessageHistory`; no history rewrite. |
| compaction | Current provider-aware threshold is 786,892 estimated tokens for V4 Flash. A committed compaction makes a stable new summary epoch instead of rebuilding the prefix every turn. | Reset warm comparison once at a compaction boundary. |
| reasoning/tool history | Complete assistant reasoning and complete tool calls are appended only after a successful provider response. | No change. |
| provider adapter | DeepSeek reports total prompt and cache-hit tokens; caching needs no request flag. | Keep raw counters; do not manufacture missing usage. |
| retry path | Retries reuse the same request, and only a complete successful response reaches history/tool execution. | No change. |
| reporting | Raw cached/input made healthy short runs look like 83-90% because call 1 is cold. | Add weighted warm reusable/reused/missed counts, hit rate, resets, and significant-miss calls. |

Offline byte-level comparison of every adjacent live request showed:

| Task | Requests | Prior messages are exact next-request prefix | Tool schemas stable |
|---|---:|---:|---:|
| none-handling | 6 | yes | yes |
| duration-parser | 6 | yes | yes |
| root-cause-cache | 6 | yes | yes |

## Implemented semantics

For a warm call in one stable request-prefix epoch:

```text
reusable = min(previous_prompt_tokens, current_prompt_tokens)
reused   = min(reusable, provider_cached_tokens)
missed   = reusable - reused
warm hit rate = sum(reused) / sum(reusable)
```

The first request is `cold_start`. Compaction or a request-prefix/schema digest
change is `reset`, not a cache miss. Missing provider cache usage is
`unavailable`, not zero. A miss is marked significant only above Pi's 1,024
token noise floor. None of these counters is an execution budget or terminal
guard.

Each `context_usage` event now includes `prompt_cache`; the final budget/eval
result includes:

- `warm_cache_reusable_tokens`
- `warm_cache_reused_tokens`
- `warm_cache_missed_tokens`
- `warm_cache_hit_ratio`
- `warm_cache_measured_calls`
- `warm_cache_significant_miss_calls`
- `warm_cache_reset_calls`

The CLI labels the original quotient `Cache/Input` and the new metric `Warm
Prefix Cache`, preventing the two meanings from being conflated.

## Ordinary live-task evidence

Run directory:
`.forgeloop/cache-audit-ordinary-20260813/20260813T050116Z-5ae2e9dc`

| Task | Result | Calls / tools | Input / cached / output | Raw cache/input | Warm reused / reusable | Warm hit | Significant misses / resets | Cost | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none-handling | PASS / `model_finish_tool` | 6 / 8 | 17,106 / 14,336 / 827 | 83.807% | 13,490 / 13,579 | 99.345% | 0 / 0 | $0.00065950 | 21.973 s |
| duration-parser | PASS / `model_finish_tool` | 6 / 8 | 19,431 / 16,256 / 1,757 | 83.660% | 14,633 / 14,968 | 97.762% | 0 / 0 | $0.00098198 | 24.471 s |
| root-cause-cache | PASS / `model_finish_tool` | 6 / 9 | 17,263 / 14,336 / 1,166 | 83.045% | 13,536 / 13,716 | 98.688% | 0 / 0 | $0.00077640 | 21.440 s |
| **weighted total** | **3/3 PASS** | **18 / 25** | **53,800 / 44,928 / 3,750** | **83.5093%** | **41,659 / 42,263** | **98.5709%** | **0 / 0** | **$0.00241788** | **67.884 s** |

The unweighted mean of the three task-level warm rates is 98.598%. Individual
misses were 0-121 tokens per call and therefore below Pi's significant-miss
floor. The 128-token-aligned provider hit counts are consistent with DeepSeek's
documented fixed-interval prefix persistence; this is an inference from the
observed counters, not a claimed provider guarantee.

## Remaining limitations

- DeepSeek caching is best-effort. A stable ForgeLoop request cannot force a
  provider hit.
- Short-run raw cached/input will remain below 98% because it correctly includes
  cold input. It must not be relabeled as warm efficiency.
- A task-specific workspace path prevents full cross-task prompt reuse. Changing
  the frozen prompt solely to improve cross-task cold-start caching is not
  justified by the current 98.5709% warm result.
- Compaction and intentional dynamic tool-schema changes start a new epoch. The
  first request after either is excluded from warm waste rather than hidden.
- Providers that omit cached-token usage yield `unavailable`; ForgeLoop does not
  substitute zero.
- The initial 98% observation used three ordinary tasks; the independent
  12-task extension below strengthens it, but still does not guarantee all
  repositories, providers, long idle gaps, or provider backend revisions.

## Stability gate extension

A second, independent batch used the completed CLI gate:

```text
--min-warm-cache-hit-rate 0.98
```

The batch selected 12 new Stage C tasks explicitly: eight easy and four medium.
No hard or DeepSWE task was selected.

| Result | Calls / tools | Input / cached / output | Provider miss | Warm reused / reusable | Warm hit | Gate | Cost | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 12/12 PASS | 73 / 97 | 205,440 / 173,440 / 11,267 | 32,000 | 161,568 / 163,800 | **98.6374%** | **PASS >=98%** | $0.00812041 | 259.177 s |

All 73 responses reported both hit and miss tokens, and all 73 satisfied
`prompt_tokens = prompt_cache_hit_tokens + prompt_cache_miss_tokens`. The run
used one backend fingerprint, with zero backend/model changes, zero compaction
or prefix resets, and zero significant misses. The maximum warm miss on any
call was exactly 128 tokens; request-start intervals were at most 11.312
seconds. This supports provider prefix-unit granularity, rather than a changing
ForgeLoop prefix, as the remaining miss source.

Across both ordinary batches, 15/15 tasks passed and weighted warm reuse was
203,227 / 206,063 = **98.6237%**. This combined figure is an offline aggregation;
the executable gate intentionally judged the second batch independently.

The adapter now preserves explicit zero usage values, records official
`prompt_cache_miss_tokens`, and exposes the DeepSeek backend
`system_fingerprint`. Previously, a provider-reported zero selected through
Python `or` could become `None`; zero hit/miss/output/reasoning values now remain
authoritative zeros.

## Hard-task canary

After the independent 12-task ordinary gate passed, one existing hard internal
task was run exactly once: `atomic-reservation`. No DeepSWE job was started.

| Result | Calls / tools | First edit / validation | Input / cached / miss / output | Warm reuse | Gate | Patch | Cost / wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| PASS / `model_finish_tool` | 6 / 8 | call 3 / call 4 PASS | 18,025 / 14,976 / 3,049 / 1,133 | 13,921 / 14,143 = **98.4303%** | PASS >=98% | 788-byte non-empty diff | $0.00078603 / 25.813 s |

All six responses satisfied provider prompt accounting, used the same backend
fingerprint, and had no significant miss; the maximum per-call warm miss was 99
tokens. The agent validated with
`python -m unittest tests.test_inventory -v`, reviewed the diff, and the
isolated official internal verifier passed. The patch first validates every
requested quantity and available stock, then mutates inventory in a second
loop, so a failure leaves stock unchanged.

## One-shot SQLfmt DeepSWE canary

After commit `c9f29fa69defab50c94dbde5905bbfd206070555` was pushed to
`main`, `sqlfmt-create-table-ddl-formatting` ran exactly once with the pinned
DeepSWE/Pier integration and the unchanged 256 model-call / 1,024 tool-call /
5,400-second policy. It was not retried.

| Result | Calls / tools | Edit / test | Input / cached / miss / output | Warm reuse | Retry | Agent termination | Patch / verifier | Cost / wall |
|---|---:|---:|---:|---:|---:|---|---|---:|
| FAIL | 25 / 29 | none / none | 908,750 / 850,944 / 57,806 / 36,834 | 830,097 / 830,255 = **99.9810%** | 0 | `provider_output_limit` | 0 bytes; F2P 0/32, P2P 1273/1273 | $0.020789 / 533.507 s |

All 25 responses satisfied provider prompt accounting, used one backend
fingerprint, and recorded no significant cache miss or prefix reset. The
largest warm miss was 77 tokens. No cumulative-token, repeated-action,
post-edit, phase-local, or short-horizon guard ended the trajectory.

On call 25 the provider returned `finish_reason=length` with 8,192 output tokens
(all reported as reasoning), no complete tool call, and a long unfinished design
monologue. ForgeLoop correctly rejected that response as completion and ended
with `provider_output_limit`. The delivery boundary produced no patch; artifact
collection then failed closed with `artifact_collection_delivery_failed`
instead of grading a fabricated or pristine patch as a successful delivery.
The official verifier therefore reported F2P 0/32 and P2P 1273/1273.

This run is evidence against further cache optimization as the next bottleneck:
warm cache was effectively perfect, while the model failed to convert analysis
into an edit before exhausting one response's configured output limit.
