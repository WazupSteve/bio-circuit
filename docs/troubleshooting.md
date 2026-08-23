# Troubleshooting

## Git LFS pointer or missing visible data

From repository root run `git lfs install`, then `git lfs pull --include='environments/circuit_supernodes/circuit_supernodes/data/visible_dataset/heldout-cohorts.jsonl.gz'`. A pointer is a tiny text file beginning `version https://git-lfs.github.com`. Use `git lfs ls-files` and `gzip -t <path>` to distinguish materialization from corruption. The other TaskSet datasets are absent by design; build them.

## TaskSet import fails

Run from `environments/circuit_supernodes`, execute `uv sync --frozen`, and check `uv run python -c 'from verifiers.v1.loaders import taskset_class; print(taskset_class("supernode-rediscovery"))'`. IDs map hyphens to importable underscore shim packages. Rebuild/reinstall the wheel after packaging changes.

## Credentials or deployment

Use `uv run modal profile current` and `uv run prime whoami`. An absent `circuit-supernodes/Attributor` means deploy `modal_app/attribution.py` in the same Modal workspace. Do not place credentials in TOML or commits.

## Provider failures and retries

Policy-provider errors and Modal errors are independent. Modal grading calls retry four times with waits of 1, 4, and 15 seconds. Repeated errors should be diagnosed rather than hidden by larger concurrency. Dataset builders report rejected sampling/graph records in triage.

## Docker selection

The shipped configs use the `null` harness, so no policy sandbox is required. For a different harness, select subprocess/Docker/Prime/Modal deliberately and account for its CPU, memory, disk, GPU, and network boundaries. Docker needs a running daemon and compatible image architecture.

## Model drift

Verify the policy model, Qwen subject revision, transcoder set, Circuit Tracer version, pruning thresholds, and provider behavior. The manifest records generation settings, but loaders do not enforce them. Regenerate rather than compare incompatible provenance as a benchmark.
