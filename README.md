# Promptless supernode discovery

A Verifiers 0.2.1 environment in which an agent discovers a requested, curated
supernode by designing its own prompts for a transcoder-equipped subject model.
Each tool call runs circuit tracer on the agent-authored prompt and returns a
label-free attribution graph. A hidden causal replacement distance is the reward.

The implementation lives in `environments/circuit_supernodes`. Evaluation and
RL configurations live in `configs/`.

## Quickstart

```bash
cd environments/circuit_supernodes
uv sync
uv run modal setup
uv run modal deploy modal_app/attribution_gemma.py

# Build one task for every pre-existing supernode in a curated circuit.
uv run python scripts/build_rediscovery_dataset.py \
    --source path/to/curated-gemma-graph.json \
    --out circuit_supernodes/data/rediscovery.jsonl

# PRIME_API_KEY is required for policy-model evaluation.
uv run eval @ ../../configs/eval/circuit-supernodes.toml --no-push
```

Agent-requested circuit traces and the hidden replacement grader use the deployed
Modal service. The initial task contains the curator's specific supernode
description and a prompt budget, but no subject prompt or graph.
The grader maps stable `layer_feature` identities to the hidden reference
position and compares the proposal and reference ablation distributions. Exact
functional equivalence scores `1`; reward halves for each configured TV-distance
half-life. The former strict pass remains a diagnostic.

See the [environment documentation](environments/circuit_supernodes/README.md)
for the task contract, reward, validation, and training workflows.
