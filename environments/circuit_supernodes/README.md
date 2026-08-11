# active promptless supernode discovery

This worktree adds the separate `supernode-rediscovery` Verifiers 0.2.1
taskset for discovering a
pre-identified semantic supernode. The agent receives the curator's specific
supernode description. It must write diagnostic prompts for a separate,
transcoder-equipped subject model, call circuit tracer on those prompts, and
infer recurring features from the returned label-free attribution graphs. A
hidden causal replacement distance is the reward.

The causal-circuit implementation inherited from the shared baseline remains in
the tree for comparison, but it is not the exported task in this worktree.

## Task contract

- Harness: `null`
- Maximum turns: 8
- Experiment budget: 7 calls to `circuit_run_ablation` by default, overridable
  per dataset row (`experiment_budget`) for budget-curve variants
- Available graph: every feature retained by Circuit Tracer's `0.8` node
  threshold, plus embedding and logit nodes
- Displayed graph: the policy chooses the feature count with `circuit_view_graph`
- Allowed proposal size: 1–8 disjoint supernodes, at least 2 features each
- Reserved terminal node: `output`

The independent promptless rediscovery task instead has:

- Subject-prompt budget: 5 by default
- Initial graph or subject prompt: none
- Tool: `circuit_run_prompt(prompt)` runs the Gemma-2-2B subject and circuit tracer
- Tool graph: up to 150 features with human-authored feature labels stripped
- Reward: `2 ** (-output_tv / replacement_reward_half_life)` for valid proposals
- Exact functional equivalence: `1.0`; each `0.05` TV halves reward by default
- Strict-pass diagnostic: same top token and at most `0.05` TV
- Diagnostics: output distance, reference/proposal effects, and membership metrics

Circuit-tracer node ids have the form `layer_feature_position`. Position changes
with each agent-authored prompt, so grading and final answers use the stable
`layer_feature` pair. Full node ids are accepted and normalized. Unknown or
malformed node syntax fails the replacement test.

The hidden grader uses the curator's original subject prompt and supernode
positions. It first zero-ablates the reference set, then substitutes the agent's
stable feature set into that same ablation experiment. Valid proposals receive
the continuous reward `2 ** (-output_tv / replacement_reward_half_life)`.
`replacement_reward_half_life` defaults to `0.05`, so an exact replacement gets
`1.0`, TV `0.05` gets `0.5`, and TV `0.10` gets `0.25`. The former strict rule—
matching argmax and TV no greater than `replacement_max_tv`—is retained as the
`replacement_pass` diagnostic. Literal precision, recall, and exact match also
remain diagnostics, so causally interchangeable IDs can earn full reward without
matching the curated membership.

Graph inspection is free and repeatable. It only changes how much of the fixed
exported graph the policy sees; the hidden response bank and scoring universe
contain every feature in that export.

The final response must contain one JSON code block:

```json
{
    "nodes": ["<layer>_<feature>", "<layer>_<feature>"]
}
```

## Inherited causal-circuit reward (not used by active discovery)

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

The backend intervenes on every exported feature and records its signed local
effect on every other exported feature plus target-token log-odds. Mediator
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
The claim is relative to Circuit Tracer's `0.8`-threshold export and the local
derivative conditions, not every hidden transformer feature or nonlinear
interaction. Lean proves the algebraic reward properties; validation of the
response bank and backend remains empirical.

## Setup

Run commands from this directory:

```bash
uv sync --group dev
uv run modal setup
uv run modal deploy modal_app/attribution_gemma.py
```

Redeploy after changing `modal_app/attribution_gemma.py`, the subject model, or the
transcoder set.

## Inherited causal-circuit datasets

The generator has six training families and a disjoint sealed evaluation split.
The evaluation split replaces one-shot addition with path traversal, modular
arithmetic, chained rules, spatial-vector composition, and Boolean circuits,
alongside two-hop recall, agreement with attractors, balanced comparison, and
weekday arithmetic. Attribution graphs are precomputed on Modal and stored with
each retained prompt.

Before attribution, Qwen is explicitly instructed to return exactly one label
from the prompt family's finite answer vocabulary without explanation. It then
greedily samples up to 50 tokens and stops at the first answer. If the answer is correct,
any narration before it is appended to the subject prompt so the attribution
target is the answer token itself. If the answer is wrong, the builder records it
as `keep_incorrect` in the triage counts but rejects the candidate without a retry.
Candidates that never emit an answer are also rejected. Manifests record the
triage counts, and retained dataset rows record their sampled text.

Build the standard family split:

```bash
uv run python scripts/build_dataset.py --n 48 --neuronpedia-labels \
    --split train \
    --out circuit_supernodes/data/train.jsonl
uv run python scripts/build_dataset.py --n 90 --seed 101 --neuronpedia-labels \
    --split eval \
    --out circuit_supernodes/data/heldout.jsonl

# Freeze the proposal-independent banks used by the MDL scorer.
uv run python scripts/precompute_response_banks.py \
    --input train.jsonl --out train-mdl.jsonl
uv run python scripts/precompute_response_banks.py \
    --input heldout.jsonl --out heldout-mdl.jsonl
```

Response banks produced by earlier versions contain only the top 60 features
and are incompatible with the complete exported universe. Rebuild both
`*-mdl.jsonl` files before evaluation or training, passing `--overwrite` when
replacing existing files.

`--neuronpedia-labels` is optional. Labels are cached in the generated JSONL and
Neuronpedia is not called during rollouts. The data directory is ignored by Git
because attribution files are large and reproducible.

## Validate and evaluate

Validate task loading without policy inference:

```bash
uv run validate supernode-rediscovery \
    --taskset.dataset-file rediscovery.jsonl \
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

## Active supernode discovery

In this worktree, curators have already identified reference supernodes inside
published circuits. Their exact descriptions (for example, `capital cities /
say a capital city`) become task descriptions, and their memberships remain
hidden grading data. The agent receives that description and a query budget. It
chooses every prompt sent to the subject model and may compare the resulting
circuit-tracer graphs before answering. It receives no seed graph, feature
labels, visible reference membership, or correctness feedback.

Build the task library from every pre-existing supernode in one or more curated
graphs, then evaluate normally:

```bash
uv run python scripts/build_rediscovery_dataset.py \
    --source https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmao6p1xd0000ypedw0qhlhob/gemma-fact-dallas-austin.json \
    --out circuit_supernodes/data/rediscovery.jsonl
uv run eval @ ../../configs/eval/circuit-supernodes.toml --no-push
```

The shipped library contains all five curator-defined supernodes in that Gemma
circuit: `capital`, `state`, `Texas`, `preposition followed by place name`, and
`capital cities / say a capital city`. The agent returns one stable
`layer_feature` set. Its continuous reward decreases exponentially with output
distance. Strict pass, output distance, intervention effects, top-token
agreement, membership diagnostics, and prompts used are all recorded.

## Inspect a trace

```bash
jq -c '{
  task: .task.data.name,
  reward: (([.rewards[]] | add) // 0),
  prompts_used: (.metrics.prompts_used // 0),
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

`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` must be linked to the pushed
environment as secrets; they are injected into hosted evaluations and training
automatically.

## Configuration

| Field | Default | Purpose |
| --- | --- | --- |
| `taskset.dataset_file` | `dataset.jsonl` | Dataset under `circuit_supernodes/data/` |
| `taskset.experiment_budget` | `7` | Default ablation budget; rows may override |
| `taskset.reward_mode` | `eval` | Compatibility field; both modes use the same MDL reward |
| `taskset.fault_rate` | `0.0` | Canary only: fraction of rollouts whose scoring raises |

Fixed safety limits allow at most 12 ablated features, 8 probe groups, and 60
probe features per interactive experiment. Each interactive experiment costs two
subject-model passes. Building the fixed response bank costs one base pass plus
four interventions per exported feature, so a 1,400-feature export costs `5,601`
passes. It is proposal-independent and should be precomputed once. A dataset row
containing `response_bank` needs no final-scoring GPU passes; missing banks are
constructed lazily and cached as a compatibility fallback.

Promptless rediscovery has its own taskset settings:

| Field | Default | Purpose |
| --- | --- | --- |
| `taskset.dataset_file` | `rediscovery.jsonl` | Curated supernode task library |
| `taskset.prompt_budget` | `5` | Subject prompts the agent may run |
| `taskset.max_features` | `150` | Maximum features returned per trace |
| `taskset.replacement_max_tv` | `0.05` | Strict-pass diagnostic TV threshold |
| `taskset.replacement_reward_half_life` | `0.05` | TV distance that halves continuous reward |
| `taskset.modal_app` | `circuit-supernodes-gemma-attribution` | Subject service |
| `taskset.modal_cls` | `GemmaAttributor` | Subject service class |

Empty prompts and prompts over 4096 characters are rejected without spending
budget. Infrastructure failures are retried and raised; successful circuit
traces spend one prompt.

## Layout

- `circuit_supernodes/circuit_supernodes.py`: task and taskset
- `circuit_supernodes/experiments.py`: interactive ablation tool
- `circuit_supernodes/graphs.py`: graph normalization, pruning, and rendering
- `circuit_supernodes/mdl.py`: pure response-bank decoder and proved MDL formula
- `circuit_supernodes/rediscovery.py`: active prompt tool and replacement grade
- `circuit_supernodes/rubric.py`: proposal validation and bank integration
- `circuit_supernodes/datagen.py`: prompt generation
- `modal_app/attribution.py`: Modal attribution, intervention, and bank service
- `modal_app/attribution_gemma.py`: active-discovery subject and circuit tracer
- `scripts/build_dataset.py`: dataset construction
- `scripts/precompute_response_banks.py`: hidden-bank precomputation
- `scripts/build_rediscovery_dataset.py`: curate hidden-reference benchmark rows
- `tests/test_mdl_reward.py`: planted-world and adversarial validation
- `tests/test_graph_view.py`: full-export loading and policy-selected graph views
- `tests/test_rediscovery.py`: active prompting and replacement-grade validation
