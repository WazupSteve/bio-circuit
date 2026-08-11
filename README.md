# Bio-circuit

A Verifiers 0.2.1 environment for evaluating and training agents to restore a
Qwen3-4B next-token distribution after a persistent hidden feature corruption.
Agents inspect the corrupted attribution graph, run bounded restoration trials,
and submit multiplicative feature interventions.

The implementation lives in `environments/circuit_supernodes`. Evaluation and
RL configurations live in `configs/`.

## Quickstart

```bash
cd environments/circuit_supernodes
uv sync
uv run modal setup
uv run modal deploy modal_app/attribution.py

# Build six training families and two held-out evaluation families.
uv run python scripts/build_dataset.py --n 48 --neuronpedia-labels \
    --families antonym analogy acronym sequence factual incontext \
    --out circuit_supernodes/data/train.jsonl
uv run python scripts/build_dataset.py --n 16 --seed 1 --neuronpedia-labels \
    --families two_hop addition \
    --out circuit_supernodes/data/heldout.jsonl

# PRIME_API_KEY is required for policy-model evaluation.
uv run eval @ ../../configs/eval/circuit-supernodes.toml --no-push
```

Dataset generation, interactive experiments, and final reward scoring use the
deployed Modal service. Generated datasets are intentionally ignored by Git and
must be rebuilt on a fresh checkout.

Training runs on Prime Lab (`prime train run configs/rl/...`), which manages the
trainer, rollout inference, and scheduling; the environment is installed there as
a wheel from the Environments Hub, and Modal credentials reach it as environment
secrets. Two canary configs under `configs/` are hard blockers that must pass
before any real run — see the environment documentation.

[FINDINGS.md](environments/circuit_supernodes/FINDINGS.md) records the audit of
the retired supernode-partition reward. It is useful historical context, but its
measurements do not describe the active restoration objective.

See the [environment documentation](environments/circuit_supernodes/README.md)
for the task contract, reward, validation, and training workflows.
