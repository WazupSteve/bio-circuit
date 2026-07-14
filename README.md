# Bio-circuit

Verifiers 0.2 multi-turn environment for supernode circuit identification on
Qwen3-4B attribution graphs. A policy can run bounded causal ablations before
submitting its final circuit.

The environment lives in `environments/circuit_supernodes`; see its README for the reward design. Ready to run eval/RL configs are in `configs/`.

## Quickstart

```bash
cd environments/circuit_supernodes
uv sync --group dev
uv run modal setup
uv run modal deploy modal_app/attribution.py   # GPU service for datagen + reward

# held-out family split: train on 6 families, eval on 2 unseen ones
uv run python scripts/build_dataset.py --n 48 --neuronpedia-labels \
    --families antonym analogy acronym sequence factual incontext \
    --out circuit_supernodes/data/train.jsonl
uv run python scripts/build_dataset.py --n 16 --seed 1 --neuronpedia-labels \
    --families two_hop addition \
    --out circuit_supernodes/data/heldout.jsonl

# eval the configured Qwen model (Prime inference; needs PRIME_API_KEY)
uv run eval @ ../../configs/eval/circuit-supernodes.toml

# run the same task with another policy model
uv run eval @ ../../configs/eval/circuit-supernodes.toml \
    --model anthropic/claude-opus-4.8
```

Dataset generation, policy-requested experiments, and final reward scoring use
Modal. Neuronpedia labels are optional and cached during dataset generation.
The generated datasets are ignored by git and must be rebuilt on a fresh
checkout. See the [environment workflow](environments/circuit_supernodes/README.md#end-to-end-workflow)
for validation, output inspection, Neuronpedia viewing, calibration, and RL.
