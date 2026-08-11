# circuit-supernodes

`circuit-supernodes` is a multi-turn Verifiers 0.2.1 environment for restoring a
suppressed Qwen3-4B response. A policy receives an attribution graph produced
under a persistent hidden corruption and submits weighted feature
interventions that recover the original response.

## Task contract

- Harness: `null`
- Maximum turns: 8
- Experiment budget: 7 calls to `circuit_run_restoration` by default, overridable
  per dataset row (`experiment_budget`) for budget-curve variants
- Displayed graph: up to 60 feature nodes, plus embedding and logit nodes
- Allowed proposal size: 1–60 unique displayed feature nodes
- Intervention weights: finite values in `[-8, 8]`

Dataset construction samples 40–60 nodes from the clean prompt's preidentified
attribution repository, samples positive amplification factors, and rejects the
sample unless the original top token changes. It then runs attribution again
with those clamps active. Only this corrupted graph is stored as the visible
graph. Every scoring and experiment call replays the same corruption before
applying the policy's proposal.

Corrupted repository nodes are retained in the 60-node visible slice even if
the corruption changes their influence rank; remaining slots are filled by the
ordinary per-node influence ranking. They are rendered identically to every
other feature and carry no corruption marker.

For clean activation `a`, hidden factor `c`, and policy factor `r`, execution
uses `a × c × r`; either omitted factor is one. Thus policy weight `1` is a true
no-op and cannot restore clean behavior. The seed, clean activations, clean graph,
hidden node identities, and hidden factors are never included in the prompt or
tool response. The JSONL retains only a small hidden replay payload (four numbers
per corrupted node), which is negligible beside the attribution graph.

The final response must contain one JSON code block:

```json
{
  "nodes": [
    {"id": "<feature_id>", "intervention_weight": 1.0}
  ]
}
```

## Reward

`output_restoration` first requires the declared original token to be top-1. If
that gate passes, it compares the intervened distribution with the unmodified
standard distribution. Let `D_r = KL(standard || restored)`:

```text
reward = 0                       if the original token is not top-1
reward = exp(-D_r)               otherwise
```

One means the submitted intervention recreates the standard distribution. The
scorer also verifies that the clean replay has the declared target at top-1 and
that the hidden corruption still changes the top token. A missing clean KL or a
failed replay invariant is an infrastructure error, never a reward-bearing
fallback.

Malformed proposals are rejected rather than repaired. Unknown or duplicate
feature ids, missing/non-finite weights, and weights outside the allowed range
score zero.

Infrastructure failures are retried with backoff and then **raise**. They are
never scored: a false zero would corrupt every advantage in its GRPO group.

`REWARD_VERSION` records which definition produced a score. Numbers from
different versions are not comparable. This remains a first-token restoration
task; it does not compare complete generated continuations.

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
sequences, factual recall, and in-context mappings. Clean attribution is used
only to form the candidate repository; the retained graph is attributed under
the accepted corruption. Dataset admission rejects corruptions that can be
solved by applying one generic inverse weight to every visible feature.

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
| `taskset.experiment_budget` | `7` | Default restoration-test budget; rows may override |
| `taskset.reward_mode` | `eval` | Compatibility field; both modes use the same reward |
| `taskset.fault_rate` | `0.0` | Canary only: fraction of rollouts whose scoring raises |

Each restoration evaluation costs one standard forward pass plus corrupted and
composed-intervention passes. Proposals may contain at most 60 features.

## Layout

- `circuit_supernodes/circuit_supernodes.py`: task and taskset
- `circuit_supernodes/experiments.py`: interactive restoration tool
- `circuit_supernodes/graphs.py`: graph normalization, pruning, and rendering
- `circuit_supernodes/interventions.py`: shared coordinate and factor contract
- `circuit_supernodes/restoration.py`: weighted-graph validation and output reward
- `circuit_supernodes/datagen.py`: prompt generation
- `modal_app/attribution.py`: Modal attribution and restoration service
- `scripts/build_dataset.py`: dataset construction
- `tests/test_restoration.py`: contract and scoring regression tests

`mdl.py`, `rubric.py`, `scripts/precompute_response_banks.py`, and
`tests/test_mdl_reward.py` are retained only to reproduce the immediately prior
MDL rollout. They are not imported by the active task or Modal deployment.
