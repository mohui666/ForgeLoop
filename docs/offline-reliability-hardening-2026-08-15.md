# Offline Reliability Hardening

Date: 2026-08-15

## Outcome

This offline-only work closes reliability gaps around Execution Closure v3,
runtime evidence, delivery, repository search, process termination, and Pier
result parsing. It used no provider request, DeepSWE LLM job, network lookup,
or external API.

## Base-aware review

The AgentLoop already fixed its progress fingerprint and patch delivery to the
Git commit present at run start. `git_diff` was still HEAD-relative, however. If
the model committed its own source change before validation, the worktree became
clean and the review tool returned `No changes.` with `review_scope=worktree`.
Closure v3 could therefore accept a review that did not contain the patch later
delivered from run base to HEAD.

The ToolRegistry now binds the stable run-start `base_head` to context-aware
tools. An unfiltered, non-cached `git_diff` compares that base directly with the
current worktree, so its tracked evidence includes model commits, staged edits,
and unstaged edits in one final-tree patch. Untracked files retain the existing
synthetic new-file evidence. Metadata records `review_base` and
`tracked_diff_layers=["base_to_worktree"]`.

A deterministic integration test performs:

```text
source edit -> model-owned Git commit -> validation PASS
-> base-to-worktree review -> explicit finish -> delivery
```

The review observation contains both committed source changes and delivery
correctly records that ForgeLoop did not create a second commit.

## Truncated command evidence

Structured `stdout_truncated` and `stderr_truncated` flags now cross the
remaining control boundaries:

- `GitPatchDelivery` rejects truncated branch, status, HEAD, staging/commit, or
  full-patch identity output instead of parsing a partial control value;
- `VerifierResult` preserves both flags without changing the official
  exit-code-driven pass/fail decision;
- Environment Foundry records runtime truncation and its own 4,000-character
  evidence excerpting.

Fault-injection coverage exercises both streams for branch, status, HEAD, and
patch identity. Collection fails closed with no patch identity when any control
output is incomplete.

## Search parity

Local repository search now runs behind a subprocess boundary. The parent owns
the timeout and can terminate a blocked filesystem read or pathological regular
expression instead of depending on cooperative checks inside the search loop. A
timeout returns no partial matches and records
`SearchResult(error="Search timed out", timed_out=True)`.

The Docker search program now applies the same path-level credential exclusions
as local search, preventing recursive root searches from returning `.env`,
credential JSON, private-key files, or `.git` content. The parity test executes
the container search program locally against a deterministic fixture, so Docker
and network access are not required.

## Follow-up runtime hardening

Local POSIX commands now start in a new session. On timeout ForgeLoop kills the
entire process group, preventing a shell grandchild from surviving the command
boundary and modifying the workspace later. The Windows `taskkill /T` path is
unchanged and now has an explicit fallback when `taskkill` itself fails.

Pier path-kind, search, and file-list consumers reject structured stdout or
stderr truncation before parsing results. They also validate the expected JSON
shape, fail closed on non-zero path probes, and filter sensitive paths from
search/list observations. Deterministic fault injection covers both streams and
freezes the existing 40,000-character Pier truncation representation.

Shell, validation, and Git inspection observations now append an explicit
warning when either runtime stream is structurally truncated. This warning is
independent of a textual truncation marker, while exit-code success semantics
remain unchanged.

## Canonical sensitive-path enforcement

Host checks and isolated search programs now derive their predicate from the
same `SENSITIVE_NAMES` and `SENSITIVE_SUFFIXES` policy. Local and Docker search
therefore cannot silently drift from `security.is_sensitive_path` when the
credential policy changes.

Pier search and file listing apply that predicate inside the remote Python
program before serializing JSON. Sensitive file names and matching source lines
do not cross the remote-to-host result boundary. PierRuntime retains a second
host-side filter as defense in depth. Cross-runtime fixtures verify the raw
remote stdout as well as the final observation.

## Durable long-horizon state

`TrajectoryStore` now serializes concurrent appends, completes short writes,
flushes each full UTF-8 JSONL record, and advances its sequence only after a
successful append. Write, flush, and optional fsync failures truncate back to
the prior file boundary; a failed rollback poisons the store so later events
cannot be appended to ambiguous evidence. `fsync_every` controls the stronger
disk boundary and `durability_policy` exposes the effective setting.
Each store also receives a per-file nonce, so two stores created for the same
run in the same second cannot merge events into one JSONL path.

Session snapshots now use a same-directory unique temporary file followed by
flush, fsync, and atomic replace. Fault injection proves that write, flush,
fsync, or replace failure preserves the previous loadable snapshot, rolls back
the in-memory `updated_at`, and leaves no temporary session data.

Offline replay and explanation can recover all complete events before a single
unterminated, malformed final JSONL record and display an explicit evidence
warning. The strict trajectory loader and dataset pipeline still reject that
record, so forensic availability does not silently admit incomplete training
data.

## Atomic reports and portable identifiers

The shared persistence layer now provides full short-write handling,
same-directory temporary files, file fsync, atomic replace, and POSIX parent
directory sync. JSONL append additionally verifies the existing newline
boundary, rolls a failed record back to the prior byte length, and reports an
explicit ambiguous-state error if rollback itself fails. A later append refuses
an incomplete tail rather than compounding it.

Eval task records use that durable JSONL boundary; Eval summaries, Dataset
indexes/manifests/exports, DeepSWE imported reports/provenance, ConfigStore, and
ModelCache use atomic publication. DeepSWE's three evaluator evidence events are
published as one atomic trajectory replacement, so collection, verifier, and
final-diff evidence cannot be split by a process failure. ModelCache also
serializes same-instance read-modify-write updates to prevent lost concurrent
manual model records.

Session and checkpoint identifiers now share one portable 1–128 character
ASCII allowlist. Path separators, absolute/drive/UNC paths, dot traversal, Git
ref metacharacters, Unicode confusables, overlong values, and Windows device
names fail before directory creation or Git mutation. Checkpoint construction
also cleans temporary index data when Git fails.

## Verification

- focused second-pass reliability tests: 34 passed, 1 skipped;
- focused third-pass reliability tests: 52 passed;
- post-review trajectory/forensic tests: 15 passed;
- full pytest after the third pass: 248 passed, 5 skipped;
- fourth-pass persistence/identifier integration tests: 151 passed;
- full pytest after the fourth pass: 352 passed, 5 skipped;
- Ruff lint (full tree) and format check (changed Python files): PASS;
- `git diff --check`: PASS;
- sdist and wheel: PASS;
- external model calls, provider usage, and API cost: zero.

## Remaining limits

- Third-party Runtime implementations can still construct `CommandResult`
  without setting structured truncation flags. Core runtimes set the flags, but
  the default remains `False` for compatibility.
- The subprocess boundary adds one Python-process startup to each local search;
  correctness and hard timeout enforcement currently take priority over that
  small fixed cost.
- Trajectory fsync defaults to `0`, so an operating-system or power failure can
  still lose the latest flushed events unless a caller selects an fsync cadence.
  The setting is constructor-level and not yet exposed in the CLI policy.
- Forensic tail recovery is intentionally limited to replay/explanation. Dataset
  curation remains strict and will reject an incomplete trajectory.
- Atomic publication prevents partial individual files, but Dataset index and
  manifest are still two separate replacement operations rather than one
  cross-file transaction.
- ModelCache locking is process-local and instance-local; independent processes
  can still race on the same cache file even though each replacement is atomic.
- A POSIX parent-directory fsync can fail after `os.replace` has succeeded. That
  correctly surfaces an uncertain durability result, but the new target may
  already be visible and cannot be rolled back without another transaction.
