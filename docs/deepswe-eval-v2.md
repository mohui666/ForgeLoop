# DeepSWE external Eval v2

ForgeLoop uses [DeepSWE](https://github.com/datacurve-ai/deep-swe) through its
official [Pier](https://github.com/datacurve-ai/pier) runner and
[Harbor task format](https://www.harborframework.com/docs/tasks). ForgeLoop does
not copy DeepSWE tasks, images, patch collection, or verifier logic. A custom
Pier Agent bridges the existing `AgentLoop` tools to the official `/app`
container; Pier then collects the committed diff and runs the official verifier
in its separate pristine environment. The result adapter writes the familiar
ForgeLoop `tasks.jsonl`, `summary.json`, trajectories, and provenance.

The prior six frozen Qwen holdout tasks remain unchanged as a fast regression
smoke set: `duration-parser`, `query-encoding`, `atomic-reservation`,
`dependency-order`, `deep-merge`, and `interval-merge`. DeepSWE Eval v2 is the
external executable evaluation and does not replace that quick local check.

## Pinned subset

The tracked manifest is
`src/forgeloop/deepswe_assets/eval-v2-subset.json`:

- DeepSWE revision: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`
- Pier: `0.3.0`
- population: all 113 task directory IDs at that revision
- selection: sort IDs, shuffle with `random.Random(20260809)`, take 20
- population checksum: `ec3ca665c98d37555cba3ab8eaa13b70365811a9e2b9327d6664ff5e7046f8c7`

The CLI refuses a revision, population checksum, generated subset, Pier
version, or task-ID mismatch. A single-task run is also restricted to the
frozen subset.

## Setup and preflight

Pier requires Python 3.12 even though ForgeLoop core still supports Python 3.10.
Install the optional dependency and clone with LF line endings on Windows:

```powershell
uv sync --extra dev --extra deepswe
git -c core.autocrlf=false clone --filter=blob:none --no-checkout `
  https://github.com/datacurve-ai/deep-swe.git `
  .forgeloop/external/deep-swe
git -C .forgeloop/external/deep-swe -c core.autocrlf=false checkout `
  435ee89ec2f2e2289f33b0da4f992f0b7b7266b9
git -C .forgeloop/external/deep-swe config core.autocrlf false

uv run forgeloop deepswe check
```

`check` verifies Docker Engine availability, the checkout and subset pins,
Pier, free disk, and LF verifier scripts. CRLF `tests/test.sh` files fail in the
Linux verifier container and are rejected before a long model run.

Each selected official task currently declares 2 CPUs, 8,192 MB memory, 20,480
MB storage, a 5,400-second Agent timeout, a 1,800-second verifier timeout, no
GPU, and no network. The local model server remains on the host; only the
DeepSWE task/verifier runs in Docker. On the validated machine, Docker Desktop
29.6.2 was available, the first task image was 3.61 GB, total Docker image usage
after validation was 4.28 GB plus 1.82 GB build cache, and 212.74 GB remained on
the drive. The 20 GB value is an official per-task provider hint rather than a
Docker Desktop disk quota; allow substantial extra room for the 20 distinct
images and shared layers.

With `qwen3.5-4b-local`, model API cost is recorded as USD 0 while real tokens
and wall time are retained. The operational costs are local GPU time, Docker
image downloads, disk, and elapsed time. The 20-task command is deliberately
serial (`--n-concurrent 1`) to stay within the laptop's memory/GPU envelope and
may take hours.

## Run

One frozen task:

```powershell
uv run forgeloop deepswe run --task katex-multicolumn-array-spans
```

The complete fixed 20-task Eval v2:

```powershell
uv run forgeloop deepswe run --policy qwen3.5-4b-local
```

Official Pier artifacts are written under `.forgeloop/deepswe-jobs/`. Mapped
ForgeLoop reports are under `.forgeloop/eval-v2-runs/`. `provenance.json`
records the upstream revision, Pier version, seed, selection method, and source
Pier job directory.

## Validated result and limitations

The clean integration run
`deepswe-eval-v2-patch-clean-20260809` completed the official task container,
ForgeLoop Agent, patch-collection phase, separate verifier, reward, trajectory,
effects, and report mapping for `katex-multicolumn-array-spans`.

The real model result was a failure: 7 model calls, 6 tool calls, 23,007 tokens,
83.05 seconds end-to-end, terminal `completed` via `model_final_message`, and no
non-empty committed patch. The official verifier preserved all 599 pre-existing
tests, passed 0/94 feature tests, and returned binary reward 0. The empty patch
is recorded honestly as `no_patch_collected`; ForgeLoop does not synthesize a
patch or reinterpret partial reward as a pass.

Known limits:

- DeepSWE expects the Agent to commit; uncommitted or absent changes do not
  appear in the official `BASE..HEAD` patch.
- ForgeLoop limits the DeepSWE `list_files` adapter to 100 paths so one large
  repository listing cannot consume the frozen 8,192-token policy context.
- Docker is mandatory for Eval v2. LocalRuntime is intentionally not a fallback
  because that would bypass the official task and verifier environments.
- Storage enforcement depends on the Pier provider; Docker's declared storage
  value is capacity guidance, not a per-container hard limit.
- This integration adds no benchmark tasks, Controller, SFT, or RL behavior.
