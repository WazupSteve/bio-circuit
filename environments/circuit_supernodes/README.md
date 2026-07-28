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
- Allowed proposal size: 2–8 disjoint supernodes, at least 2 features each
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

### Hidden split-half separation

Scoring does not ablate a supernode whole. A seed derived as
`HMAC(salt, graph ‖ nonce)` splits every supernode into two halves; each half is
ablated separately, and the effect credited to that supernode is the geometric
mean of the two halves' fraction-normalized drops. A group therefore earns credit
only when its causal role is *distributed* across its members — which is what the
word supernode is supposed to mean — and an agent cannot pre-measure whole-group
effects and transcribe them, because it does not know the split.

The nonce comes from the run the rollout belongs to (`run.id:run.step` in
training, a constant during evaluation), so every rollout of a GRPO group is
scored under the same split by construction, the split rotates between steps so
no fixed split can be memorized, and evaluation stays reproducible. Set
`CIRCUIT_SPLIT_SALT` to commit a run's salt in advance and publish it afterwards.

`MIN_GROUP_SIZE` is 2 because a singleton has no halves — a mechanical floor, not
an adversarial one. A larger floor looks like it should make the split harder to
guess (a group of `m` admits only `C(m, m//2)/2` halvings), but the oracle audit
measured otherwise: raising it from 2 to 6 left the adaptive coherence-matched
attack no worse off and *weakened* the penalty on blind small groups. Knowing the
split only helps an attacker that can also satisfy it, which takes per-feature
measurement rather than small groups.

### Gates

Malformed proposals are **rejected, not repaired**: silently fixing them teaches
the policy that malformed proposals are fine. A proposal scores zero when it has
fewer than 2 or more than 8 supernodes, repeats or overlaps members, names a
non-FEATURE id, contains a supernode below the minimum size, gives one group over
70% of grouped influence, or whose edges do not form a DAG in which every
supernode reaches `output`.

Edges are scored as **reachability**, not as direct connections: two edge sets
with the same transitive closure receive the same score.

### Reward modes

`taskset.reward_mode = "train"` adds a small structured floor —
`0.01·parse + 0.02·membership + 0.02·topology + 0.95·causal` — so a policy that
cannot yet earn causal credit still has gradient toward a valid proposal.
`"eval"` (the default) is strict, so checkpoint selection and any published
number see the real score.

Infrastructure failures are retried with backoff and then **raise**. They are
never scored: a false zero would corrupt every advantage in its GRPO group.
Identical interventions are cached within the scorer.

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

# Or on Prime-managed infrastructure, once the environment is pushed:
prime eval run ../../configs/eval/circuit-supernodes.toml --hosted --follow
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

Reference scores for the keyword-matched circuit and a matched-random control:

```bash
uv run python scripts/calibrate.py --dataset heldout.jsonl
```

Red-team the reward before trusting it. The audit scores scripted attacks —
blind balanced groups, a community partition, random multistart, minimum-size
groups, and a coherence-matched partition built from per-feature measurements —
against both reward versions, giving each attack the true effect matrix and the
best edge fit it supports. **v2 must beat every attack; if
it does not, the reward is not measuring circuit discovery.**

```bash
uv run python scripts/oracle_audit.py --dataset heldout.jsonl --rows 8 \
    --out /tmp/oracle.json

# Does the minimum-size gate earn its keep? (It does not — see the reward notes.)
uv run python scripts/oracle_audit.py --min-group-size 6 --rows 10
```

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
| `taskset.reward_mode` | `eval` | `train` adds the structured floor |
| `taskset.fault_rate` | `0.0` | Canary only: fraction of rollouts whose scoring raises |

Fixed safety limits allow at most 12 ablated features, 8 probe groups, and 60
probe features per experiment. Each experiment costs two subject-model passes;
final scoring costs `2k + 2` passes for `k` supernodes, since both halves of
every group are ablated.

## Layout

- `circuit_supernodes/circuit_supernodes.py`: task and taskset
- `circuit_supernodes/experiments.py`: interactive ablation tool
- `circuit_supernodes/graphs.py`: graph normalization, pruning, and rendering
- `circuit_supernodes/rubric.py`: intervention-based reward
- `circuit_supernodes/datagen.py`: prompt generation
- `modal_app/attribution.py`: Modal attribution and intervention service
- `scripts/build_dataset.py`: dataset construction
- `scripts/calibrate.py`: reference scores for expected and random proposals
- `scripts/oracle_audit.py`: adversarial audit of the reward (v1 vs v2)
- `scripts/reward_components.py`: per-component means from archived traces
- `scripts/export_neuronpedia.py`: proposal export
