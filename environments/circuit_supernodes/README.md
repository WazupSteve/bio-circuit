# circuit-supernodes

`circuit-supernodes` is a multi-turn Verifiers 0.2.1 environment for circuit
discovery in Qwen3-4B. A policy model receives a pruned attribution graph,
groups feature nodes into conceptual supernodes, and predicts causal edges
between those groups.

## Task contract

- Harness: `null`
- Maximum turns: 8
- Experiment budget: 7 calls to `circuit_run_ablation` by default, overridable
  per dataset row (`experiment_budget`) for budget-curve variants
- Displayed graph: up to 60 feature nodes, plus embedding and logit nodes
- Allowed proposal size: 1–8 disjoint supernodes, at least 2 features each
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

`circuit_faithfulness` is an exponential transform of normalized
minimum-description-length saving. First compute the signed value:

```text
net_reward = 1 - circuit_error/null_error
               - 2·ln(2)·description_bits/null_error
reward = 2 ** (net_reward - 1)
```

- `null_error` is the whitened squared error of predicting zero on the entire
  fixed hidden response tensor.
- `circuit_error` is held-out error when the submitted partition and DAG form a
  block-constant predictor of that tensor.
- `description_bits` is the canonical literal encoding of the selected members,
  group assignments, and submitted edges.

The factor `2·ln(2)` is not tuned. Unit-variance Gaussian residuals cost
`error / (2·ln(2))` bits. A circuit scores above the `0.5` break-even point
exactly when its predictive bit saving exceeds the bits required to describe
it. Every valid finite circuit has positive reward; invalid proposals retain
the `0.0` sentinel. The exponential is strictly monotone, so it preserves every
ordering from signed net MDL value without collapsing net-cost circuits into a
single clipped score.

### Fixed hidden response bank

The backend intervenes on every displayed feature and records its signed local
effect on every other displayed feature plus target-token log-odds. Mediator
columns are weighted by squared output sensitivity, following the first-order
output propagation approximation. Proposal-independent scales whiten each
column.

Block values are fitted on calibration intervention radii and evaluated only on
disjoint validation radii. When predicting one source member, its calibration
row is excluded. Randomly mixed members must therefore predict one another's
held-out causal roles rather than fitting a proposal-specific experiment.

Unsubmitted source rows, unsubmitted destination columns, and unreachable blocks
predict zero. Omitting most of the mechanism cannot shrink the validation set.
The response bank is cached and can be precomputed into the hidden dataset, so
every sibling proposal is evaluated against exactly the same tensor.

### Literal circuit code

For `N` available features, `n` selected features, `K` groups and `m` edges:

```text
L = b_n + b_K + n·(b_node + b_group) + b_m + 2·m·b_endpoint
```

Each width is `ceil(log2(cardinality))`; `K_max` is 8. Feature identifiers are
sorted and group labels are canonical. Natural-language names and block values
are not encoded: names are annotations, while block values are reconstructed
from the shared calibration bank. The deliberately literal edge code ensures
that every extra member or edge costs positive bits.

### Gates

Malformed proposals are **rejected, not repaired**. A proposal scores zero when
it has fewer than 1 or more than 8 supernodes, repeats or overlaps members, names
a non-FEATURE id, contains a supernode below the leave-one-member-out minimum of
2, duplicates an edge, or does not form a DAG. There is no dominance, coverage,
or preferred-size gate; valid catch-alls and tiny circuits fail through error or
description length.

Edges are scored as **reachability**, not as direct connections. Edge sets with
the same transitive closure have the same prediction error. Redundant transitive
edges still cost additional description bits, so the shortest representative
wins.

### Reward modes

`train` and `eval` now return the same objective. V2's separately weighted
parse/membership/topology training floor was removed because it reintroduced an
arbitrary multi-reward rubric. The result also reports signed `net_reward` and
`net_bits` as diagnostics. The harness reward is its exponential transform in
`(0, 1]` for valid proposals; `0.5` is the net-saving threshold.

Infrastructure failures are retried with backoff and then **raise**. They are
never scored: a false zero would corrupt every advantage in its GRPO group.
Identical response banks are cached within the scorer.

`REWARD_VERSION` records which definition produced a score. Numbers from
different versions are not comparable. The old three-factor score is retained
only in `FINDINGS.md` as an audit record.

### What the reward does not measure

The score identifies equivalence under its frozen intervention operator, not a
human semantic name. Reachability-equivalent direct DAGs are not identifiable.
The claim is relative to the displayed feature slice and the local derivative
conditions, not every hidden transformer feature or nonlinear interaction. Lean
proves the algebraic reward properties; validation of the response bank and
backend remains empirical.

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

# Freeze the proposal-independent banks used by the MDL scorer.
uv run python scripts/precompute_response_banks.py \
    --input train.jsonl --out train-mdl.jsonl --max-features 60
uv run python scripts/precompute_response_banks.py \
    --input heldout.jsonl --out heldout-mdl.jsonl --max-features 60
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
uv run pytest
uv run ruff check .
uv run eval @ ../../configs/eval/circuit-supernodes.toml \
    --dry-run true --no-push
uv run eval @ ../../configs/eval/circuit-supernodes.toml --no-push

# Or on Prime-managed infrastructure, once the environment is pushed:
prime eval run ../../configs/eval/circuit-supernodes.toml --hosted --follow
```

Override the policy model with `--model`. Completed runs are written beneath
`outputs/`.

## Inspect a trace

```bash
jq -c '{
  task: .task.data.name,
  reward: (([.rewards[]] | add) // 0),
  experiments_used: (.metrics.experiments_used // 0),
  turns: ([.nodes[] | select(.sampled == true)] | length),
  errors
}' outputs/.../traces.jsonl
```

## Train

For hosted RL, install the Prime CLI separately and run the repository config:

```bash
uv tool install prime
prime login
cd ../..
prime train run configs/rl/circuit-supernodes.toml
```

Two canaries are hard blockers and must pass first — see
`configs/eval/canary-egress.toml` (does environment code reach the Modal backend
from a hosted run at all?) and `configs/rl/canary-error-semantics.toml` (does a
scoring exception exclude the rollout from its group, or contribute a zero?).

`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` must be linked to the pushed
environment as secrets; they are injected into hosted evaluations and training
automatically.

## Configuration

| Field | Default | Purpose |
| --- | --- | --- |
| `taskset.dataset_file` | `dataset.jsonl` | Dataset under `circuit_supernodes/data/` |
| `taskset.max_features` | `60` | Maximum feature nodes shown to the policy |
| `taskset.experiment_budget` | `7` | Default ablation budget; rows may override |
| `taskset.reward_mode` | `eval` | Compatibility field; both modes use the same MDL reward |
| `taskset.fault_rate` | `0.0` | Canary only: fraction of rollouts whose scoring raises |

Fixed safety limits allow at most 12 ablated features, 8 probe groups, and 60
probe features per interactive experiment. Each interactive experiment costs two
subject-model passes. Building the default fixed response bank costs one base
pass plus four interventions per displayed feature (`241` passes for 60
features), but it is proposal-independent and should be precomputed once. A
dataset row containing `response_bank` needs no final-scoring GPU passes; missing
banks are constructed lazily and cached as a compatibility fallback.

## Layout

- `circuit_supernodes/circuit_supernodes.py`: task and taskset
- `circuit_supernodes/experiments.py`: interactive ablation tool
- `circuit_supernodes/graphs.py`: graph normalization, pruning, and rendering
- `circuit_supernodes/mdl.py`: pure response-bank decoder and proved MDL formula
- `circuit_supernodes/rubric.py`: proposal validation and bank integration
- `circuit_supernodes/datagen.py`: prompt generation
- `modal_app/attribution.py`: Modal attribution, intervention, and bank service
- `scripts/build_dataset.py`: dataset construction
- `scripts/precompute_response_banks.py`: hidden-bank precomputation
- `tests/test_mdl_reward.py`: planted-world and adversarial validation
