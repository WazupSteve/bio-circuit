# Bio-circuit environments

This repository contains independent Verifiers 0.2.1 tasksets for causal-circuit
MDL modeling and active promptless supernode discovery. Shared graph,
serialization, validation, and retry primitives live in
`environments/circuit_supernodes`.

Evaluation and RL configurations live in `configs/` and are named after their
task contract.

## Quickstart

```bash
cd environments/circuit_supernodes
uv sync
uv run modal setup
uv run modal deploy modal_app/attribution.py
uv run modal deploy modal_app/attribution_gemma.py

# Build causal MDL datasets.
uv run python scripts/build_dataset.py --n 48 --neuronpedia-labels \
    --split train --out circuit_supernodes/data/train-mdl.jsonl
uv run python scripts/build_dataset.py --n 72 --seed 1 --neuronpedia-labels \
    --split eval --out circuit_supernodes/data/heldout-mdl.jsonl

# Rebuild promptless rediscovery data from a curated source graph.
uv run python scripts/build_rediscovery_dataset.py \
    --source path/to/curated-gemma-graph.json \
    --out circuit_supernodes/data/rediscovery.jsonl

# PRIME_API_KEY is required for policy-model evaluation.
uv run eval @ ../../configs/eval/circuit-mdl.toml --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --no-push
```

The compact rediscovery task library is checked in. Generated causal MDL
datasets are intentionally ignored and must be rebuilt on a fresh checkout.

Read [FINDINGS.md](environments/circuit_supernodes/FINDINGS.md) for the causal
reward audit, and see the
[environment documentation](environments/circuit_supernodes/README.md) for the
task contracts, validation, and deployment workflows.
