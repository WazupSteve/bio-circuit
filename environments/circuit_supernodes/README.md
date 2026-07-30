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
70% of grouped influence, or whose edges do not form a DAG.

At least one supernode must reach `output`, but not every one. A group with no
outgoing edge claims that ablating it changes nothing downstream, which is a
falsifiable prediction the calibration term already checks. Requiring every group
to reach `output` made the true answer unstateable whenever a group is genuinely
off-path — the honest proposal was rejected and the agent had to assert an effect
the subject model does not have, which cost the correct answer about 14%.

Edges are scored as **reachability**, not as direct connections: two edge sets
with the same transitive closure receive the same score.

### Held-out intervention operator

`circuit_run_ablation` clamps features to zero. Scoring **mean-ablates**,
removing only each feature's deviation from its own average activation over the
prompt. Fitting edges to whatever the agent measured is what lets any partition
score well, so the operator the agent can buy is deliberately not the operator it
is graded under, and a circuit has to generalise across them to earn credit.

The two disagree about ordering, not just magnitude — on a sample graph zero-clamp
ranks one group as moving the target most (0.906 vs 0.875) while mean-ablation
ranks the other (0.860 vs 0.838) — and reordering is what breaks a fitted edge
set. A uniform rescaling would not do this: it attenuates every feature equally
and fitted edges transfer unchanged.

### Reward modes

`taskset.reward_mode = "train"` adds a small structured floor —
`0.01·parse + 0.02·membership + 0.02·topology + 0.95·causal` — so a policy that
cannot yet earn causal credit still has gradient toward a valid proposal.
`"eval"` (the default) is strict, so checkpoint selection and any published
number see the real score.

Infrastructure failures are retried with backoff and then **raise**. They are
never scored: a false zero would corrupt every advantage in its GRPO group.
Identical interventions are cached within the scorer.

`REWARD_VERSION` records which definition produced a score. Numbers from
different versions are not comparable — the frontier ranking under v1 turned out
to be an artefact of terms that v2 removed.

### What the reward does not measure

The displayed slice is the top `max_features` by influence, and the influence
distribution is flat: 60 features carry only about 31% of total feature
influence, half of it would take roughly 196 features and 80% roughly 722. The
complement ablation touches only the ungrouped *displayed* features, so the rest
is never intervened on. A proposal at coverage 1.0 has grouped everything it was
shown, which is a considerably weaker claim than having captured the circuit.

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
