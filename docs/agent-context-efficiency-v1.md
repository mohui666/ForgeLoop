# Agent Context Efficiency v1

Cache-stability follow-up: [Provider Reliability v1 and Pi agent-core audit](provider-reliability-v1-2026-08-13.md).

Agent Context Efficiency v1 bounds repeated model input in long `AgentLoop`
trajectories. It does not change Controller v1.3 Simplified, V4-Flash, tool
availability, DeepSWE tasks, patch collection, verifiers, or total budgets.

## Baseline audit

The audit used the actual `model_request` snapshots and provider-reported usage
from the three frozen Controller v1.3 DeepSWE trajectories. Character counts are
used only to attribute repeated request content; token totals are the provider's
measurements.

| Task | Provider input | Largest repeated sources |
|---|---:|---|
| `sqlfmt-create-table-ddl-formatting` | 136,871 | tool observations 259,115 chars; old reasoning 117,362 chars |
| `sqlite-utils-safe-import-checkpoints` | 197,316 | old reasoning 271,073 chars; tool observations 267,447 chars |
| `textual-kitty-key-phases` | 192,236 | tool observations 382,409 chars; old reasoning 95,655 chars |

Those two sources represented 72%–77% of attributed repeated characters. Within
tool observations, `read_file` accounted for 70%–78%; older repository listings,
searches, and shell output supplied most of the remainder. Tool schemas,
tool-call arguments, system prompts, and Controller recovery messages were
smaller but repeated on every request.

## Deterministic policy

The canonical in-memory history and append-only trajectory keep the full
evidence. Only the message list sent on the next provider request is prepared:

- the system/task prefix is retained verbatim;
- the latest three action turns stay complete;
- successful patch calls and observations remain, with obsolete historical
  reasoning removed;
- reads of edited files and the latest two source reads remain complete;
- the latest diff, test, failed test, and tool error remain complete;
- older reads, searches, listings, shell observations, and Controller feedback
  become a deterministic provenance ledger;
- the ledger retains tool name, arguments (including path/range, search pattern,
  and shell command), status/exit evidence, and bounded output/error excerpts.

Compaction starts only when the serialized request estimate reaches 6,000
tokens and only when it makes the request smaller. There is no summarizer model,
supervisor, classifier gate, or additional model call.

## Per-call observability

Every completed provider call appends a `context_usage` event using schema
`forgeloop.context.v1`. It records provider input/cache tokens, pre/post
compaction estimates, retained/compacted turn counts, source character totals,
tool-observation totals by tool, and the largest sources. `model_request` also
records the pre-call context report, so provider failures retain the estimate.

`forgeloop trace explain <trajectory>` prints every call under `Model input
context`, including actual input, estimate, compaction delta, and three largest
sources.

## One-shot validation on 2026-08-09

The unchanged `query-encoding` task passed once with 6 model calls, 8 tool calls,
15,398 input tokens, 1,757 output tokens, and explicit edit/test/finalization.
Its per-call input remained below the long-trajectory threshold.

Each of the same three DeepSWE tasks was then run exactly once with pinned
DeepSWE revision `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`, Pier 0.3.0,
V4-Flash, Controller v1.3 Simplified, and the unchanged official verifier.

| Task | Input before -> after | Calls before -> after | Edit / validation / finalization after | Terminal after | Verifier |
|---|---:|---:|---|---|---:|
| `sqlfmt-create-table-ddl-formatting` | 136,871 -> 81,092 (-40.8%) | 10 -> 7 | no / no / no | failed / `controller_no_change_final` | FAIL |
| `sqlite-utils-safe-import-checkpoints` | 197,316 -> 184,983 (-6.3%) | 14 -> 18 | yes (4 patches) / no / no | budget exceeded / `budget_guard` | FAIL |
| `textual-kitty-key-phases` | 192,236 -> 184,943 (-3.8%) | 13 -> 15 | yes (1 patch) / no / no | budget exceeded / `budget_guard` | FAIL |

Across the three tasks, input fell from 526,423 to 451,018 (-14.3%) and total
tokens fell from 579,642 to 502,371 (-13.3%). Model calls increased from 37 to
40 because the two edit-producing tasks spent the saved context budget on more
steps before reaching the unchanged 200,000-token guard. On their last calls,
the effective estimates were 29.5%, 68.9%, and 49.2% below the uncompressed
requests, respectively.

The result demonstrates materially smaller and bounded per-call context without
restoring the old pre-edit Controller bottleneck: two tasks still edited. It does
not demonstrate a solve. No task produced a collector patch, all official
verifiers failed, and the new `sqlite-utils` sample did not repeat the baseline's
post-edit validation before budget exhaustion. Total-budget efficiency therefore
remains a separate problem from context growth.

## Verification

The implementation passed:

```powershell
uv run pytest
uv run ruff check .
uv build
```
