# Operations

## Local checks and evaluation

Run nested commands from `environments/circuit_supernodes`:

```bash
uv sync --frozen
uv run pytest
uv run eval @ ../../configs/eval/visible-dataset.toml --dry-run --no-push
```

`eval` is Verifiers v1 local orchestration. A real rollout calls the configured policy provider and the deployed Modal subject, writes under `outputs/`, retries Modal calls up to four attempts, and may cost money. `--no-push` disables result upload, not provider calls.

## Modal deployment

From `environments/circuit_supernodes`, after budget approval:

```bash
uv run modal deploy modal_app/attribution.py
uv run modal run modal_app/attribution.py --help
```

The `Attributor` runs on an A100-80GB, has a one-hour timeout, up to eight containers, and a ten-minute scale-down window. Model assets use a persistent Hugging Face cache volume. Use `uv run modal app stop circuit-supernodes` to stop the deployment; confirm the current Modal CLI syntax before teardown.

## Self-hosted GPU (no Modal)

See [local.md](local.md): `scripts/subject.sh up` brings up a GPU host, serves the subject, and tunnels it; `export SUBJECT_URL=http://localhost:8000` switches every client off Modal.

## Hosted publication and evaluation

Hosted evaluation is a separate workflow. Publish each TaskSet environment under an owner slug, publish its required dataset, then invoke Prime's hosted evaluation interface with that published identifier. The local Verifiers-v1 TOMLs in `configs/eval/` are not claimed to work with `prime eval run --hosted`.

## Dataset generation

From `environments/circuit_supernodes`, these dispatch remote Modal work and write local generated data plus a provenance sidecar:

```bash
uv run python scripts/build_visible_dataset.py --help
uv run python scripts/build_promptless_dataset.py --help
uv run python scripts/build_restoration_dataset.py --help
```

Choose `--out` filenames matching [reference](reference.md). Generation samples prompts deterministically but calls nondeterministic models/services; partial provider failures are triaged. Generated multi-GB data is intentionally not committed unless owners approve it.

## Reinforcement learning

Training is advanced and paid. It requires generated train/eval datasets, published environments, provider and Prime credentials, and explicit cost approval. From repository root:

```bash
prime train configs/rl/visible-dataset.toml
prime train configs/rl/supernode-rediscovery.toml
prime train configs/rl/suppressed-features.toml
```

Validate configs locally before dispatch; do not run training in pull-request CI.

## Manual remote acceptance

With budget approval: deploy staging; run one sample and attribution; build one task for each mode; complete one rollout per TaskSet. Test hosted evaluation and RL only after corresponding environments and datasets are published. Record model/transcoder versions and tear down staging resources.
