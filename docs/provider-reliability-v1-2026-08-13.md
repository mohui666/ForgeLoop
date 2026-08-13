# Provider Reliability v1 and Pi agent-core audit

Date: 2026-08-13

## Scope and frozen boundaries

This change makes one incomplete model request resilient to transient provider or
transport failure, then applies the reusable agent-core lessons found in Pi. It
does not change V4-Flash generation settings or the `execution-first-v1` prompt,
Controller policy, Execution Budget v2 horizon, Context Efficiency evidence
selection, DeepSWE tasks/verifiers/collector, or any training path.

The Pi comparison used `badlogic/pi-mono` revision
`581d75a89cea21e50d6a26df840352f94427f633`. The relevant upstream contracts are
Pi's [provider retry implementation](https://github.com/badlogic/pi-mono/blob/581d75a89cea21e50d6a26df840352f94427f633/packages/ai/src/utils/retry.ts),
[agent loop](https://github.com/badlogic/pi-mono/blob/581d75a89cea21e50d6a26df840352f94427f633/packages/agent/src/agent-loop.ts),
and [compaction design](https://github.com/badlogic/pi-mono/blob/581d75a89cea21e50d6a26df840352f94427f633/packages/coding-agent/docs/compaction.md).
The audit covered request lifecycle, streaming completion, retry and cancellation,
context/cache behavior, tool-call validation, command-output truncation, sessions,
usage/cost, and execution ordering.

## DeepSeek cache and task-result baseline

`cached/input` is now emitted directly as `cached_input_ratio`; it is cache
telemetry, not benchmark solve rate. Before the cache-stability repair, the
available Provider Reliability canaries showed:

| Run | Complete responses | Input | Cached | Cached/input |
| --- | ---: | ---: | ---: | ---: |
| query-encoding | 6 | 16,648 | 13,952 | 83.81% |
| SQLfmt | 67 | 1,793,063 | 715,136 | 39.88% |
| SQLite snapshot at 91 responses | 91 | 2,592,637 | 1,427,200 | 55.05% |

The weighted ratio for that snapshot is about 49.0%. This does not imply a 49%
task pass rate: query-encoding was the only completed official result at that
point; SQLfmt was externally interrupted and SQLite was still running.

Trajectory inspection identified a deterministic cache-breaker. The old 6,000
estimated-token trigger rebuilt the provenance ledger on almost every long-horizon
turn. SQLfmt reached 95.2% cached/input at call 4, then compaction began at call 5
and the next calls fell to 10.7% and 6.4%.

ForgeLoop now follows Pi's boundary shape: compact near
`context_window - reserve_tokens`, with a conservative estimator margin, commit
one summary/ledger epoch, and keep that request prefix byte-stable until another
epoch is actually needed. The append-only canonical history remains separate for
audit. V4-Flash resolves to a 786,892 estimated-token threshold with the current
1,000,000-token capability and 8,192-token generation configuration. Every
request records the effective threshold and compaction epoch.

## Provider Reliability v1

### Request boundary and accounting

- One model call is one logical request. Physical retry attempts do not increment
  model-call budget, append another assistant turn, or execute any tool.
- ForgeLoop disables LiteLLM/OpenAI hidden SDK retries (`max_retries=0`,
  `num_retries=0`) so attempt telemetry and backoff are attributable.
- Only a complete successful response is appended to history and charged to
  usage/cost. Failed attempts without provider usage record
  `usage.status=unavailable`; they are never filled with zero.
- If a later logical request ultimately fails without usage, earlier successful
  counters remain available as the known accumulated amount. The summary marks
  `usage_complete=false`, records `usage_records`, and counts
  `unavailable_model_calls` instead of erasing or pretending to complete them.
- A later model request can be retried without replaying a tool call that already
  completed on an earlier logical request.

### Classification

Retryable evidence:

- explicit 408, 409, 429, and 5xx status;
- timeout and connection exception classes;
- SSL EOF/record failures, incomplete stream/chunk/read, reset/abort/disconnect,
  broken pipe, and explicit overloaded/gateway/server-unavailable messages.

Permanent evidence:

- authentication, permission, bad request, not found, unprocessable request,
  context-window, content-policy, and unsupported-parameter classes;
- other 4xx responses;
- insufficient balance/quota and billing hard limits, even when surfaced as 429;
- unknown errors without affirmative transient evidence.

Explicit permanent SDK classes take precedence over incidental three-digit
numbers in exception text.

### Default policy

| Parameter | Default |
| --- | ---: |
| maximum attempts including the first | 4 |
| retry count | 3 |
| initial backoff | 1.0 s |
| multiplier | 2.0 |
| maximum backoff | 8.0 s |
| jitter | +/-20% |
| per-attempt timeout cap | 600 s |

The backoff is bounded by the remaining trajectory wall clock and is
user-interruptible. The effective object is stored in policy identity, trajectory
events, DeepSWE metadata, task results, and `provenance.json` under schema
`forgeloop.provider-reliability.v1`.

### Streaming and limited-output safety

Streams are consumed and validated inside the provider/client lifetime. A stream
must contain a terminal finish reason and assemble into a complete assistant
message. Partial content and incomplete tool calls are discarded before history or
tool execution. Provider-reported usage is accepted only from the completed
stream; LiteLLM builder estimates are not labeled provider usage.

Pi also treats tool arguments from an output-token-limited response as unsafe even
when salvaged JSON parses. ForgeLoop now rejects every tool call in a `length` or
equivalent response, returns an error observation asking the model to reissue it,
and performs no side effect. Content-filter/safety-limited responses use the same
fail-closed boundary. A limited final response without tools terminates as
`provider_output_limit` or `provider_safety_limit`, not false completion.

Malformed JSON tool arguments are retained in assistant history for protocol
continuity but become an execution-blocked observation. Structural tool-schema
errors (required fields, top-level type/enum, unexpected fields) are likewise
model-recoverable. Bounds such as `minItems` and `minLength` remain tool-owned so
Controller tools keep their existing semantic recovery and terminal behavior.

## Other Pi audit decisions

| Area | Decision |
| --- | --- |
| command output | Preserve both head and tail around a deterministic omission marker so terminal diagnostics survive. |
| read pagination | Existing `start_line`/`end_line` remains the continuation contract; no second file protocol was added. |
| cancellation | Retry backoff is interruptible; LocalRuntime already kills the Windows process tree on timeout and Docker uses an in-container hard timeout. |
| tool ordering | Keep sequential execution. Pi's parallel independent tools were not copied because ForgeLoop batches can contain source-changing and state-dependent calls. |
| sessions | Keep the existing redacted session summary plus full trajectory/checkpoints. Pi's branching session tree is not required for the current single-workspace product boundary. |
| steering/planner | Not added. The frozen prompt/Controller and one-loop architecture remain authoritative. |
| usage/cache | Preserve provider input/cached/output/reasoning/cost independently and expose cached/input; no token counter becomes an execution stop. |

## Deterministic evidence

Fault-injection tests cover:

1. two SSL EOF failures followed by success on the same logical request;
2. incomplete stream followed by success, with no partial tool execution;
3. retryable 5xx followed by success;
4. permanent 4xx immediate failure;
5. retry exhaustion to `provider_failure`;
6. a previously completed tool call not duplicated by a later request retry;
7. cancellation during backoff;
8. billing/quota 429 fail-fast behavior;
9. output-limited tool rejection and output-limited final fail-closed behavior;
10. malformed JSON and structural schema failures as recoverable observations;
11. cache-stable committed compaction with append-only canonical history;
12. command output retaining both first and final diagnostics.

## Live canaries

Both DeepSWE tasks used the frozen revision
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier revision
`34c18f0e4eed88877c28721f5c5871a950bec637`, the official unchanged collector and
verifier, and the 256 model-call / 1,024 tool-call / 5,400-second horizon.

| Task | Calls / tools | First edit / first validation | Retry result | Usage / cost / wall | Terminal / collector / verifier |
| --- | --- | --- | --- | --- | --- |
| SQLfmt | 68 requests, 67 responses / 130 tools | none / none | no natural failure | 1,793,063 input / 715,136 cached / 19,377 output; successful-response cost $0.15833772; 653.664 s recorded trajectory span | external host/session interruption while request 68 was in flight; no ForgeLoop terminal or official result |
| SQLite | 196 logical calls / 197 provider attempts / 260 tools | edit call 64 / validation call 144 (80-call gap); 23 validations, 12 PASS | call 139 attempt 1 failed, attempt 2 recovered | 11,368,342 input / 7,753,984 reported cached / 213,067 output; $0.58738004; 2,648.304 s end-to-end | `model_finish_tool`; 38,806-byte collector patch; official PASS, F2P 60/60 and P2P 1038/1038 |

The final-code `query-encoding` regression also passed 3/3 official tests:
10 model calls, 12 tools, first edit call 3, first validation call 4,
36,176 input / 32,384 cached / 2,228 output (89.52% cached/input),
$0.00124540, 38.663 seconds, no retry, and `model_finish_tool`.

The SQLfmt run is intentionally not converted into a model/provider failure: its
trajectory ends after `provider_attempt_started` because the host Docker/session
was reset. The failed physical request had no returned usage and the run did not
reach Pier collection.

The SQLite canary naturally encountered a DeepSeek 500/SSL record-layer failure
on logical model call 139. Attempt 1 recorded unavailable usage, scheduled
1.054920 seconds of jittered backoff, and attempt 2 completed successfully. The
same trajectory continued; no assistant/tool history was duplicated.

One of the 196 successful SQLite responses did not report `cached_tokens`.
Therefore the exact cumulative cached count is unavailable; 7,753,984 is the
honest reported lower bound from the other 195 responses (68.21% of total input),
and the mapped task field remains `null` rather than treating the missing value as
zero. Input, output, and cost were present on all successful responses.

Delivery committed a 38,805-byte base-to-HEAD patch from base
`8d74ffc93292c604d5827e2b44fffedca0c28c19`; the official collector emitted the
canonical newline-terminated 38,806-byte `model.patch`. The fail-closed audit
matched its content/provenance, and the clean verifier explicitly applied 38,806
bytes before running 1,050 repository tests plus the 60 F2P tests.

This long process started before the final 600-second per-attempt cap was loaded,
so its attempt telemetry shows the then-effective remaining-wall timeout
(`3585.094s`) on the recovered request. The retry boundary, classifier, backoff,
and history/accounting semantics were active. The final 600-second cap is covered
by policy/provenance tests and applies to newly started runs; this canary was not
restarted merely to refresh that one configuration value.

## Quality gates

- `pytest`: PASS, 171 tests, including the Docker/Pier deterministic patch
  collection fixture;
- Ruff: PASS;
- `uv build`: PASS for `forgeloop-0.1.0.tar.gz` and
  `forgeloop-0.1.0-py3-none-any.whl`;
- `git diff --check`: PASS;
- final-code `query-encoding`: PASS, 3/3 verifier tests.

## Remaining provider and agent-core risks

- A synchronous SDK call depends on the SDK/transport honoring its timeout; Python
  cannot safely kill an arbitrary blocked thread. The final policy caps one
  attempt at 600 seconds and the independent 5,400-second horizon still applies.
- A provider may charge a failed request without returning usage. ForgeLoop records
  that attempt as unavailable instead of inventing cost.
- Streaming currently buffers chunks until completion for atomic validation. Very
  large streamed outputs can use substantial memory; the frozen V4 policy remains
  non-streaming.
- Local tool cancellation is guaranteed at command timeout, not instantaneously
  while a synchronous tool call is in progress. Completed effects remain durable.
- Command output preserves head and tail but does not yet persist the omitted full
  output as a separate artifact.
- Cache ratios depend on each provider's token-field semantics. ForgeLoop reports
  the raw provider counters and the transparent cached/input quotient without
  claiming cross-provider equivalence.
