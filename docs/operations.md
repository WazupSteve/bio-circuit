# Operations

Commands run from `environments/circuit_supernodes` unless noted.

## Evaluate

```bash
uv run eval @ ../../configs/eval/visible-dataset.toml --dry-run --no-push   # validate only
uv run eval @ ../../configs/eval/visible-dataset.toml --num-tasks 1 --no-push
```

`eval` is Verifiers 0.3 local orchestration. A run:

- sends policy requests to Prime Inference (`model` in the config, or `--model`);
- runs the `circuit-null` harness in a Prime sandbox, connected to the circuit
  tools by a tunnel;
- calls the subject backend from the tools and graders on your machine;
- writes `outputs/<run>/traces.jsonl` and logs.

Re-run missing or failed rollouts with `--resume`, or start over with
`--clean`. `--no-push` disables uploading results only.

## Deploy the Modal subject

```bash
uv run modal deploy modal_app/attribution.py
uv run modal run modal_app/attribution.py --prompt "The capital of France is"   # smoke test
```

The `Attributor` class runs on A100-80GB, with a one-hour call timeout, up to
eight containers and a ten-minute idle window. Weights cache in the
`circuit-supernodes-hf-cache` volume. The GPU image (`modal_app/backend/deployment.py`)
pins `circuit-tracer`, `torch`, `transformer-lens` and `pyarrow<26`
(transformer-lens needs NumPy 1.x, which pyarrow 26 dropped).

Stop idle spend:

```bash
uv run modal container list                 # find circuit-supernodes containers
uv run modal container stop -y <container-id>
uv run modal app stop circuit-supernodes    # stop the whole app (undeploys)
```

In a shared workspace, stop only containers that belong to `circuit-supernodes`.

## Self-hosted subject

`scripts/subject.sh up` provisions a GPU host, serves the subject and tunnels
it; `export SUBJECT_URL=http://localhost:8000` switches clients to it. See
[local.md](local.md).

## Build datasets

```bash
uv run python scripts/build_visible_dataset.py --help
uv run python scripts/build_promptless_dataset.py --help
uv run python scripts/build_restoration_dataset.py --help
```

Builders call the deployed backend and write data plus a manifest. See
[data generation](data-generation.md); use the file names in
[reference](reference.md).

## Train

The RL configs are templates. They expect training files that are not
released and a reachable subject backend. From the repository root:

```bash
prime train configs/rl/visible-dataset.toml
prime train configs/rl/supernode-rediscovery.toml
prime train configs/rl/suppressed-features.toml
```

They follow the Verifiers 0.3 layout but have not been validated with
`prime train`.

## Publish

The Prime Hub holds release 0.1.x (`docs/public-release.json`): core
`circuit-supernodes` 0.1.1, plus 0.1.2 distributions of `supernode-rediscovery`,
`suppressed-features` and `circuit-null` that pin the core version. Those
packages predate the Verifiers 0.3.1 migration. Republishing needs a version
bump in `environments/circuit_supernodes/pyproject.toml` and matching pins in
`environments/{circuit_null,supernode_rediscovery,suppressed_features}/pyproject.toml`.

## Acceptance check for a deployment

1. Run the Modal smoke test above.
2. Run one task per taskset with `--num-tasks 1`.
3. In each trace, confirm `baseline_drift = 0` and `grading_validation_error = 0`,
   and that tools returned results.
4. Stop leftover containers.
