# Quickstart

Run these commands from a local checkout. The evaluation client uses Python
3.13; the Modal deployment defines its own GPU image.

## 1. Install

```bash
cd environments/circuit_supernodes
uv sync --frozen
```

The default evaluation datasets download from Hugging Face on first use. No
Hugging Face token or Git LFS setup is required. The three compressed datasets
total about 2.6 GiB; allow additional RAM for parsed attribution graphs.

## 2. Connect to Modal

Use credentials for the workspace containing the `circuit-supernodes` app:

```bash
uv run modal setup
```

For an existing deployment, continue to the next step. To deploy the included
backend in your own workspace:

```bash
uv run modal deploy modal_app/attribution.py
```

The backend loads Qwen3-4B and its transcoders on A100-80GB GPUs. It can scale to
eight containers, so GPU usage is billed to the deployment's Modal workspace.

Leave `SUBJECT_URL` unset to use Modal. If it was set for a local GPU server:

```bash
unset SUBJECT_URL
```

For automated runs, provide `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` through
your runner's secret configuration. Keep credentials out of config files and Git.

## 3. Configure policy inference

The supplied configs use Prime Inference. Set `PRIME_API_KEY` in your shell or
runner's secret configuration. The policy model is selected by `model` in each
TOML file; Qwen3-4B remains the subject being studied.

## 4. Check the configuration

```bash
uv run eval @ ../../configs/eval/visible-dataset.toml --dry-run --no-push
```

This validates configuration without calling the policy or GPU backend.

## 5. Run an evaluation

```bash
uv run eval @ ../../configs/eval/visible-dataset.toml --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --no-push
uv run eval @ ../../configs/eval/suppressed-features.toml --no-push
```

Each config selects 16 tasks and one rollout per task. These commands call the
policy provider and Modal backend. `--no-push` keeps evaluation results local;
it does not disable inference or GPU charges.

See [operations](docs/operations.md) for hosted execution and training,
[troubleshooting](docs/troubleshooting.md) for common failures, and
[data generation](docs/data-generation.md) to build additional cohorts.
