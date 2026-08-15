# Execution Closure v3

Date: 2026-08-15

## Outcome

Execution Closure v3 removes a deterministic premature terminal from the
long-horizon V4-Flash path. After validation and worktree review, the model now
retains the remaining Execution Budget v2 horizon and must explicitly call
`finish` (or return an accepted final response). The Controller no longer ends
the run merely because one unchanged model decision followed a review.

The new policy identity is
`deepseek-v4-flash-controller-v1.5-explicit-closeout`. It preserves the model,
revision, `execution-first-v1` prompt, tool set, Context Efficiency, 32,000-token
Pi-style output window, provider reliability settings, and configurable
execution budget. The v1.4 policy and Closure v2 Controller remain bundled for
historical replay.

## Saved SQLfmt evidence

The 2026-08-13 SQLfmt canary did not voluntarily finish. Its late trajectory
was:

```text
call 93  repository unit + functional tests PASS (1271)
call 94  end-to-end tests PASS (40)
call 95  manual edge-case probes PASS
call 96  git_diff(path="src/sqlfmt/ddl.py")
         observation contained status only for the untracked file
call 97  read_file("src/sqlfmt/ddl.py") to continue final review
terminal controller_ready_auto_finish
```

The path-filtered `git_diff` did not display the new file's content, but Closure
v2 treated the tool name alone as a complete review. On the next call the model
was still reviewing the implementation when the Controller terminated it. The
official verifier later exposed formatting idempotency and return-type contract
failures. Hidden tests cannot be predicted deterministically, but the harness
must not cut off an active review before the model chooses to finish.

## Semantics

Closure v3 changes only closeout evidence and termination:

- a path-filtered `git_diff`, cached-only diff, shell diff, or failed diff is
  not complete worktree-review evidence;
- the structured unfiltered `git_diff` request records `review_scope=worktree`;
- staged and unstaged tracked patches are both shown, in application order;
- untracked UTF-8 source is rendered as a new-file diff, while binary/non-UTF-8
  files are represented by size and SHA-256;
- truncated tracked or untracked output is partial evidence and cannot complete
  the review gate;
- sensitive tracked or untracked content is withheld and the review remains
  incomplete;
- ForgeLoop trajectories and other non-deliverable cache artifacts are excluded
  using the same worktree semantics as delivery;
- quoted and Git-magic path names use shell-aware literal pathspecs;
- validation and review remain bound to the exact source fingerprint;
- edits still invalidate validation and review and require retesting;
- after complete review, reads, probes, and further reasoning remain allowed;
- only an explicit model finish/final response, the real execution horizon,
  provider failure, cancellation, or another retained safety terminal can end
  the trajectory.

The change is generic. It does not inspect task names, verifier tests, SQL
syntax, or benchmark-specific paths.

## Deterministic coverage

Tests cover:

- partial path review does not advance `needs_review`;
- unfiltered worktree review advances to `ready_to_finish`;
- call 50 with an unchanged reviewed tree still does not auto-finish;
- explicit `finish` is accepted after current-tree validation and review;
- an accepted plain final response is also a valid explicit decision;
- the integration path performs edit -> validate -> review -> additional read
  -> explicit finish -> delivery;
- untracked file content is present in `git_diff` output;
- staged changes are included, while tracked and untracked sensitive content is
  withheld and cannot produce complete review;
- structured truncation metadata prevents even markerless truncation from being
  treated as complete;
- cached-only and failed diffs cannot satisfy the review gate;
- DeepSeek Plan mode remains read-only and does not install the execution
  Controller;
- v1.4 continues to instantiate Closure v2, while v1.5 records Closure v3.

No model or provider request is required for this validation.

## Verification

- focused Controller/tool/policy/closure tests: PASS;
- full pytest: 191 passed, 4 skipped;
- Ruff lint: PASS;
- sdist and wheel: PASS, with the v1.5 policy manifest present in the wheel;
- `git diff --check`: PASS.

The previously established `query-encoding` PASS was not rerun because this
implementation session explicitly prohibited model tests. No API call, retry,
usage, or cost was produced.
