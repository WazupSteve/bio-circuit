# circuit-supernodes

`circuit-supernodes` is a multi-turn Verifiers 0.2.1 environment for circuit
discovery in Qwen3-4B. A policy model receives a pruned attribution graph,
groups feature nodes into conceptual supernodes, and predicts causal edges
between those groups.

## Task contract

- Harness: `null`
- Maximum turns: 8
- Experiment budget: 7 calls to `circuit_run_ablation`
- Displayed graph: up to 60 feature nodes, plus embedding and logit nodes
- Allowed proposal size: 2–8 disjoint supernodes
- Reserved terminal node: `output`

Each experiment clamps selected feature nodes to zero and reports the change in
the target-token probability and optional probe-group activations. Experiments
provide evidence but do not directly contribute reward.

The final response must contain one JSON code block:

```json
{
  "supernodes": [
    {"name": "<concept>", "nodes": ["<feature_id>"]}
  ],
  "edges": [["<source>", "<target-or-output>"]]
}
```

## Reward

The `circuit_faithfulness` reward performs independent interventions on the
subject model:

```text
reward = separation^0.6 × coverage^0.2 × completeness^0.2
```

- `separation` measures whether predicted downstream groups change more than
  unpredicted groups and whether the effect sizes are calibrated.
- `coverage` is the share of displayed feature influence included in the
  proposal.
- `completeness` measures how well the selected circuit accounts for the target
  token after the ungrouped complement is ablated.

The reward is gated to zero when a proposal has fewer than two valid
supernodes, one group holds more than 70% of grouped influence, a supernode has
no outgoing edge, or the edge set makes no falsifiable prediction. Identical
interventions are cached within the scorer.

## Setup

Run commands from this directory:

```bash
uv sync --group dev
uv run modal setup
uv run modal deploy modal_app/attribution.py
uv run modal run modal_app/attribution.py \
    --prompt "Fact: Michael Jordan plays the sport of"
```

Redeploy after changing `modal_app/attribution.py`, the subject model, or the
transcoder set.

## Datasets

The generator supports two-hop recall, addition, antonyms, analogies, acronyms,
sequences, factual recall, and in-context mappings. Attribution graphs are
precomputed on Modal and stored with each retained prompt.

Build the standard family split:

```bash
uv run python scripts/build_dataset.py --n 48 --neuronpedia-labels \
    --families antonym analogy acronym sequence factual incontext \
    --out circuit_supernodes/data/train.jsonl
uv run python scripts/build_dataset.py --n 16 --seed 1 --neuronpedia-labels \
    --families two_hop addition \
    --out circuit_supernodes/data/heldout.jsonl
```

`--neuronpedia-labels` is optional. Labels are cached in the generated JSONL and
Neuronpedia is not called during rollouts. The data directory is ignored by Git
because attribution files are large and reproducible.

## Validate and evaluate

Validate task loading without policy inference:

```bash
uv run validate circuit-supernodes \
    --taskset.dataset-file heldout.jsonl \
    --runtime.type subprocess --only-setup true --rich false -n 1
```

Validate the evaluation configuration, then run it:

```bash
uv run ruff check .
uv run eval @ ../../configs/eval/circuit-supernodes.toml \
    --dry-run true --no-push
uv run eval @ ../../configs/eval/circuit-supernodes.toml --no-push
```

Override the policy model with `--model`. Completed runs are written beneath
`outputs/`.

## Inspect and export a proposal

Inspect a trace:

```bash
jq -c '{
  task: .task.data.name,
  reward: (([.rewards[]] | add) // 0),
  experiments_used: (.metrics.experiments_used // 0),
  turns: ([.nodes[] | select(.sampled == true)] | length),
  errors
}' outputs/.../traces.jsonl
```

Export one proposal for Neuronpedia:

```bash
sed -n '1p' outputs/.../traces.jsonl > /tmp/rollout.json
uv run python scripts/export_neuronpedia.py \
    --dataset heldout.jsonl --index 0 --proposal /tmp/rollout.json \
    --slug heldout-0 --out /tmp/heldout-0.json
```

Upload the output at <https://www.neuronpedia.org/graph/validator>. The dataset
index must match `task.data.idx` when evaluation is not shuffled.

## Calibrate and train

Calibration uses live Modal interventions:

```bash
uv run python scripts/calibrate.py --dataset heldout.jsonl
```

For hosted RL, install the Prime CLI separately and run the repository config:

```bash
uv tool install prime
prime login
cd ../..
prime rl run -c configs/rl/circuit-supernodes.toml
```

Run a small evaluation and inspect its reward distribution before training.

## Configuration

| Field | Default | Purpose |
| --- | --- | --- |
| `taskset.dataset_file` | `dataset.jsonl` | Dataset under `circuit_supernodes/data/` |
| `taskset.max_features` | `60` | Maximum feature nodes shown to the policy |

Fixed safety limits allow at most 12 ablated features, 8 probe groups, and 60
probe features per experiment. Each experiment costs two subject-model passes;
final scoring costs approximately one pass per supernode plus two additional
passes.

## Layout

- `circuit_supernodes/circuit_supernodes.py`: task and taskset
- `circuit_supernodes/experiments.py`: interactive ablation tool
- `circuit_supernodes/graphs.py`: graph normalization, pruning, and rendering
- `circuit_supernodes/rubric.py`: intervention-based reward
- `circuit_supernodes/datagen.py`: prompt generation
- `modal_app/attribution.py`: Modal attribution and intervention service
- `scripts/build_dataset.py`: dataset construction
- `scripts/calibrate.py`: reward calibration
- `scripts/export_neuronpedia.py`: proposal export
