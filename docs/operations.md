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

The Prime Hub holds release 0.2.1 under `wazupsteve` (`docs/public-release.json`):
the core `circuit-supernodes` package and three thin aliases,
`supernode-rediscovery`, `suppressed-features` and `circuit-null`, each pinning
the exact core version.

To publish a new release:

1. Bump `version` in `environments/circuit_supernodes/pyproject.toml`, and the
   version and `circuit-supernodes==` pin in
   `environments/{circuit_null,supernode_rediscovery,suppressed_features}/pyproject.toml`.
2. From the repository root, push the core first, then the aliases:

   ```bash
   uv run prime env push --path environments/circuit_supernodes --owner wazupsteve --visibility PUBLIC
   uv run prime env push --path environments/supernode_rediscovery --owner wazupsteve --visibility PUBLIC
   uv run prime env push --path environments/suppressed_features --owner wazupsteve --visibility PUBLIC
   uv run prime env push --path environments/circuit_null --owner wazupsteve --visibility PUBLIC
   ```

3. In a fresh venv, install all four with `uvx --from prime prime env install`
   (core first) and check that the three taskset IDs and the harness resolve.

An alias's Hub index doesn't serve the core package, so an alias installed on
its own fails to resolve; the core must already be installed. Running `prime`
inside the target venv is also unsafe: installing the core upgrades `click`
and breaks that venv's `prime` CLI, so use `uvx`.

## Acceptance check for a deployment

1. Run the Modal smoke test above.
2. Run one task per taskset with `--num-tasks 1`.
3. In each trace, confirm `baseline_drift = 0` and `grading_validation_error = 0`,
   and that tools returned results.
4. Stop leftover containers.
