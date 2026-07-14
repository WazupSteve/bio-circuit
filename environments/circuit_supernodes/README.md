# circuit-supernodes

### Overview
- **Environment ID**: `circuit-supernodes`
- **Short description**: Identify supernodes and circuits in pruned attribution graphs of Qwen3-4B, computed with per-layer transcoders via circuit-tracer on Modal.
- **Tags**: multi-turn, interpretability, train, eval

### Datasets
- **Primary dataset(s)**: Synthetic circuit-eliciting prompts sampled from the prompt families in Anthropic's [attribution-graphs methods paper](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (two-hop factual recall, addition, antonyms/multilingual, analogies, acronyms, sequences, factual recall, in-context pattern matching). Each prompt's pruned attribution graph is precomputed on a Modal GPU with [circuit-tracer](https://github.com/safety-research/circuit-tracer) and cached into `circuit_supernodes/data/dataset.jsonl`.
- **Generation**: `scripts/build_dataset.py` creates a held-out-family split:
  six families for training and the structurally richer `two_hop` and `addition`
  families for evaluation. Unfaithful prompts are dropped.
- **Feature labels**: `--neuronpedia-labels` optionally fetches explanations for
  the 60 displayed Qwen3-4B transcoder features in one batch request per graph.
  Labels are cached in the JSONL; Neuronpedia is never called during rollouts.
- **Split sizes**: configurable; default 48 train / 16 held-out

### Task
- **Type**: multi-turn Verifiers 0.2 `Task` / `Taskset` with the `null` harness
- **Interaction**: up to three `circuit_run_ablation` tool calls, then a final
  proposal. Each experiment accepts features to clamp and optional feature
  groups to probe; it reports output-probability and probe-activation changes.
- **Output format**: short analysis + one JSON code block: `{"supernodes": [{"name", "nodes"}], "edges": [[src, tgt-or-"output"]]}`
- **Reward**: a single causal signal, `circuit_faithfulness` (GPU inference at scoring
  time via the deployed Modal app). Each proposed supernode — and the complement of
  everything left ungrouped — is ablated in the actual model (circuit-tracer
  `feature_intervention`, attention frozen):

  `reward = separation^0.6 * coverage^0.2 * completeness^0.2`

  - **separation** — `sqrt(rank_skill * calibration_skill)` over relative
    activation/probability drops: a soft-sigmoid AUC ranking drops predicted by the
    proposal's transitive edge closure above unpredicted ones, and a Brier
    calibration macro-averaged per ablated supernode (so failed predictions can't
    hide behind another group's easy wins). Both are skill scores — chance-level
    evidence earns exactly 0.
  - **coverage** — influence share of grouped features over the total shown.
  - **completeness** — log-odds preservation of the target token when all ungrouped
    features are ablated at once.

  Experiments do not directly add reward; they provide evidence for the final
  proposal. Gated to exactly 0: fewer than 2 valid supernodes, a catch-all supernode holding
  >70% of grouped influence, a supernode with no outgoing edge, or edges that
  predict everything/nothing (nothing falsifiable). Identical proposals are cached,
  so repeated rollouts are free.

  Calibrate the reward's dynamic range against reference proposals (datagen's
  expected circuit, random grouping, and the degenerate top-drivers+decoy hack):
  `uv run python scripts/calibrate.py --dataset heldout.jsonl`

  Measured baselines over all 20 held-out rows: hack **0.00** (always gated),
  random grouping **0.24** mean. Off-the-shelf models score **0.54–0.68** mean
  on the 16-example eval config (16k max tokens). The `expected` reference
  currently scores 0 because its keyword matcher can't bridge tokenization
  (e.g. "36" vs digit tokens `3`,`6`) — a known limitation of the weak
  reference, not of the reward.

### End-to-end workflow

Run every command in this section from `environments/circuit_supernodes` unless
stated otherwise.

#### 1. Install and authenticate

```bash
cd environments/circuit_supernodes
uv sync --group dev
uv run modal setup                 # once per machine
export PRIME_API_KEY=...           # policy-model evaluation
```

#### 2. Deploy and smoke-test the subject model

```bash
uv run modal deploy modal_app/attribution.py
uv run modal run modal_app/attribution.py \
    --prompt "Fact: Michael Jordan plays the sport of"
```

Redeploy after changing `modal_app/attribution.py`, the subject model, or the
transcoder set.

#### 3. Build the train and held-out datasets

```bash
uv run python scripts/build_dataset.py --n 48 --neuronpedia-labels \
    --families antonym analogy acronym sequence factual incontext \
    --out circuit_supernodes/data/train.jsonl
uv run python scripts/build_dataset.py --n 16 --seed 1 --neuronpedia-labels \
    --families two_hop addition \
    --out circuit_supernodes/data/heldout.jsonl
```

Omit `--neuronpedia-labels` for a fully API-independent build. If the optional
request fails, dataset generation continues with blank labels. Data files are
gitignored and must be rebuilt on a fresh checkout.

Inspect the generated rows:

```bash
wc -l circuit_supernodes/data/{train,heldout}.jsonl
jq -c '{family, prompt, target}' circuit_supernodes/data/heldout.jsonl
```

#### 4. Validate and evaluate

```bash
uv run pytest
uv run ruff check .
uv run eval @ ../../configs/eval/circuit-supernodes.toml --dry-run true --no-push
uv run eval @ ../../configs/eval/circuit-supernodes.toml --no-push
```

Override the policy model with `--model`, for example:

```bash
uv run eval @ ../../configs/eval/circuit-supernodes.toml \
    --model anthropic/claude-opus-4.8 --no-push
```

Completed runs are written under `outputs/`. Locate and inspect a trace file:

```bash
find outputs -name traces.jsonl -print
jq -c '{example_id, reward, error}' outputs/.../traces.jsonl
jq . outputs/.../traces.jsonl | less -R
```

#### 5. View a proposal in Neuronpedia

Save one rollout, combine it with the corresponding dataset graph, and upload
the resulting JSON:

```bash
sed -n '1p' outputs/.../traces.jsonl > /tmp/rollout.json
uv run python scripts/export_neuronpedia.py \
    --dataset heldout.jsonl --index 0 --proposal /tmp/rollout.json \
    --slug heldout-0 --out /tmp/heldout-0.json
```

Upload `/tmp/heldout-0.json` at
<https://www.neuronpedia.org/graph/validator>. The `--index` must match the
rollout's `example_id` when evaluation is not shuffled.

#### 6. Calibrate or train

```bash
uv run python scripts/calibrate.py --dataset heldout.jsonl
```

Calibration uses live Modal interventions. For hosted RL, install the Prime CLI
and launch the repository config from the repository root:

```bash
uv tool install prime
prime login
cd ../..
prime rl run -c configs/rl/circuit-supernodes.toml
```

Run a small evaluation first and inspect its reward distribution before
starting training.

### Environment configuration

Verifiers 0.2 uses typed `[taskset]` and `[harness]` sections. The fixed
four-turn contract allows three experiment turns plus one final-answer turn.

| Taskset config field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_file` | str | `"dataset.jsonl"` | Which cached dataset file under `circuit_supernodes/data/` to load |
| `max_features` | int | `60` | Feature nodes kept in the slimmed graph shown to the model |

Safety limits are fixed at 3 experiments per rollout, 12 ablated features,
8 probe groups, and 60 total probe features per call.

### Architecture
- `circuit_supernodes/circuit_supernodes.py` — typed task, taskset, and reward
- `circuit_supernodes/experiments.py` — bounded interactive ablation tool
- `circuit_supernodes/datagen.py` — synthetic prompt sampler with expected conceptual circuits (weak ground truth, used by `scripts/calibrate.py`)
- `modal_app/attribution.py` — Modal GPU service for graph attribution and
  feature interventions used by both experiments and final scoring
- `circuit_supernodes/graphs.py` — graph normalization, influence shares, slimming, in-context rendering
- `circuit_supernodes/rubric.py` — `InterventionScoring`: the causal `circuit_faithfulness` reward
- `scripts/export_neuronpedia.py` — export a proposal as a Neuronpedia-ready graph JSON
- `tests/` — offline tests for parsing, graph handling, dataset generation, export, and every reward gate (Modal stubbed)

### Cost tradeoff

Each interactive experiment costs two subject-model passes: baseline and
ablation. With the default budget, a rollout can add six passes before final
scoring, which still costs roughly `k + 2` passes for `k` proposed supernodes.
