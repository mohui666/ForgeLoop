# DeepSWE / Pier Patch Collection & Delivery

Date: 2026-08-12

This repair changed only patch delivery, collection qualification, and result
provenance. It made no model calls and did not change V4-Flash, the Controller,
prompt, execution budget, Context Efficiency, DeepSWE tasks/verifiers, or any
training code.

## Root cause

ForgeLoop pins DeepSWE at
`435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`. Its parent
`d7a1031eec255f92d05d6ed9b23b638c57ab741b` removed every task's
`pre_artifacts.sh` and replaced it with `[[verifier.collect]]`. For ABS the
official command is:

```sh
cd /app && mkdir -p /logs/artifacts && \
  git config --global --add safe.directory /app && \
  git diff --binary cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a HEAD \
  > /logs/artifacts/model.patch
```

The released PyPI `datacurve-pier==0.3.0` predates that contract. It knows the
older optional `pre_artifacts.sh` path but has no `VerifierCollectConfig` and no
`Trial._run_collect_hooks`. Because the current task has no script, the script
discovery was a no-op. Pydantic ignored the unknown `verifier.collect` field,
so no command materialized the patch.

The old ABS trial proves the behavior rather than merely the version mismatch:

- agent cwd/repository: `/app` at base
  `cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a`, branch `master`;
- delivery branch: `forgeloop/deepswe-delivery-cb1b3b67`;
- delivery HEAD: `0231abd07781fb648498acae772da0f513c49d3b`;
- delivery result: committed, clean, 27,843-byte non-empty normalized patch;
- artifact manifest: `/logs/artifacts` `empty`, explicit `model.patch` `failed`;
- trial log: Docker could not find `/logs/artifacts/model.patch`;
- verifier output: `no model.patch submitted`, so it graded pristine base.

The delivery commit existed in the live agent container and was collector
visible until Pier stopped/deleted that environment. The missing execution
hook, not ForgeLoop Git visibility, broke the chain.

## Compatibility fix

The optional dependency and lockfile now pin Pier repository
`https://github.com/datacurve-ai/pier.git` at
`34c18f0e4eed88877c28721f5c5871a950bec637`. This is the coordinated Pier
commit that added `VerifierCollectConfig`, parsing of `[[verifier.collect]]`,
and execution immediately before artifact collection; DeepSWE switched its
tasks three minutes later. The package metadata still reports `0.3.0`, so
ForgeLoop no longer accepts a version string alone.

`forgeloop deepswe check` now requires all of the following:

1. exact Pier repository and Git commit from `direct_url.json`;
2. installed code that parses the collector hook;
3. `/logs/artifacts/model.patch` declared by every frozen task;
4. a collector command using that task's metadata base SHA through `HEAD`.

Delivery provenance now includes the SHA-256 of the normalized patch. During
result import ForgeLoop audits the task base, delivery base/head/branch,
delivery bytes/hash, artifact manifest, collected bytes/hash, and the official
verifier's `model.patch applied (N bytes)` evidence. Empty/missing artifacts,
wrong base, delivery mismatch, manifest failure, apply failure, or missing
apply evidence become `infrastructure_failed` with an explicit
`artifact_collection_*` reason. They cannot be counted as model failures or
successful submissions.

## Actual execution chain

For the pinned DeepSWE tasks the fixed Pier flow is:

1. ForgeLoop runs in the agent container at `/app` and captures the base HEAD.
2. `GitPatchDelivery` commits a dirty tree, or adopts a commit already made by
   the model, and verifies a clean non-empty `base..HEAD` patch.
3. Pier checks `pre_artifacts.sh`; current DeepSWE v1.1 tasks have none, so this
   compatibility step is a no-op.
4. Pier executes `[[verifier.collect]]` in the still-live main agent container.
   The command itself changes cwd to `/app` and writes the official
   base-to-HEAD diff to `/logs/artifacts/model.patch`.
5. Pier records the artifact manifest and transfers the patch to the host.
6. Pier stops the agent environment, creates the separate verifier image from
   the task's `tests/Dockerfile`, and uploads the artifact.
7. The official grader in pristine `/app` resets touched paths to the declared
   base, applies `model.patch`, applies hidden `test.patch`, and grades.
8. ForgeLoop imports the reward only after the fail-closed artifact audit.

## Deterministic end-to-end evidence

`tests/test_deepswe_patch_collection.py` uses no model. Its custom deterministic
Pier agent creates a real source file, calls the production
`GitPatchDelivery`, and commits it in the agent container. The pinned Pier hook
then collects the patch and a separate clean verifier checks and applies it.

Observed result:

| Evidence | Result |
|---|---:|
| model calls / token usage / API cost | 0 / 0 / USD 0 |
| delivery patch bytes (normalized) | 171 |
| collected `model.patch` bytes | 172 |
| artifact manifest | directory `ok`, file `ok` |
| clean verifier `git apply --check` | PASS |
| clean verifier observed source content | PASS |
| Pier reward | 1.0 |

Unit coverage also retains both delivery paths: uncommitted dirty-tree delivery
and a commit created before delivery. Completed empty patches, wrong bases,
unapplyable patches, and collection failures are rejected.

## Offline ABS recovery

The deleted old agent container made its commit object unavailable, but its
trajectory retained 19 successful `apply_patch` edits. Replaying them against a
clean copy of the official image at the exact base reconstructed five changed
files. Two parser edits contained trajectory secret-redaction markers; their
original Go token expressions were recovered from the pristine preimage.

The reconstructed `git diff --binary base` is 27,846 bytes on disk and 27,843
bytes after the same `.strip()` normalization used by the old delivery event,
an exact match to its recorded byte count. It applies in an independent clean
worktree. The official unmodified DeepSWE verifier then reported:

- `model.patch applied (27846 bytes)`;
- P2P 6/6, F2P 5/6, partial 0.9166666667;
- binary reward 0 because the hidden parser test expected `[:5:2]` while the
  patch rendered `[0:5:2]`.

Thus the old patch is recoverable and replayable offline. The original pristine
score was invalid evidence about the patch; correct grading still yields FAIL,
but for one real behavioral defect rather than six untouched feature tests.
Because the old delivery event did not record a hash, exact historical byte
identity cannot be proven cryptographically. New deliveries record that hash.

## Remaining limitations

- Pinned Pier supports collect hooks only on the main service; sidecar-service
  hooks are skipped.
- Pier itself treats hook/download errors as best effort. ForgeLoop's importer
  is the fail-closed boundary; consumers reading raw Pier reward alone must
  also inspect the artifact manifest.
- Installation of the DeepSWE extra needs Python 3.12 and access to the pinned
  Git commit (or a populated package cache).
- The result JSON renders patch bytes as UTF-8 text for convenience. The
  authoritative `model.patch` file and SHA-256 provenance remain byte based,
  including for binary diffs.
- Historical runs without `patch_sha256` can be checked only by base, byte
  count, replay/apply evidence, and verifier output.
- A verifier can still fail for genuine model behavior after a correct patch
  lifecycle; infrastructure qualification does not reinterpret that failure.
