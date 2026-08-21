# Quickstart

This builds randomly mixed evaluation datasets for the three task modes and
runs hosted Kimi K3 rollouts against them.

## 1. Install the environment

From the repository root:

```bash
uv sync
cd environments/circuit_supernodes
uv sync
```

Configure the credentials required by Modal and Prime before continuing.

## 2. Deploy the shared subject-model backend

All three modes use the same Modal `Attributor` deployment:

```bash
uv run modal deploy modal_app/attribution.py
```

## 3. Generate mixed-family evaluation datasets

Visible circuit discovery:

```bash
uv run python scripts/build_visible_dataset.py \
  --n 2400 \
  --tasks 16 \
  --seed 1 \
  --split eval \
  --out circuit_supernodes/data/visible_dataset/heldout-cohorts.jsonl.gz
```

Promptless supernode discovery:

```bash
uv run python scripts/build_promptless_dataset.py \
  --n 480 \
  --tasks 16 \
  --seed 1 \
  --split eval \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes.jsonl.gz
```

Suppressed-feature restoration:

```bash
uv run python scripts/build_restoration_dataset.py \
  --n 480 \
  --tasks 16 \
  --seed 1 \
  --split eval \
  --out circuit_supernodes/data/output_restoration/suppressed-features.jsonl.gz
```

The builders randomly mix completed family rows using `--seed`, so the same
seed reproduces the same ordering.

## 4. Run hosted Kimi K3 evaluations

Return to the repository root:

```bash
cd ../..
```

Visible circuit discovery:

```bash
uv run prime eval run configs/eval/visible-dataset.toml \
  --model moonshotai/Kimi-K3 \
  --num-examples 16 \
  --rollouts-per-example 4 \
  --hosted
```

Promptless supernode discovery:

```bash
uv run prime eval run configs/eval/supernode-rediscovery.toml \
  --model moonshotai/Kimi-K3 \
  --num-examples 16 \
  --rollouts-per-example 4 \
  --hosted
