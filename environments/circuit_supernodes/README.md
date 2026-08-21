# Circuit task contracts

The package exports three independent Verifiers TaskSets from one core package. Model-node-count is a
universal presentation policy, not a fourth plugin: each dataset stores the
maximal Circuit Tracer export (up to 4,096 feature nodes), and the policy chooses
an ordinary top-k view at inference time. Required, goal, corrupted, or held-out
nodes are never automatically inserted into those views.
Dataset rows are deliberately runtime-minimal. Their sidecar manifests record
the graph policy, exporter settings, sampler configuration, schema version, and
corruption provenance once per artifact.

## Shared answer sampling and families

Every generated task calls the same bounded greedy sampler before attribution:

- the subject is instructed to emit one answer from `possible_answers`;
- generation stops at the first complete answer label, within 50 tokens;
- correct matches are retained, wrong matches are counted and rejected, and
  exhaustion/errors are rejected;
- narration before the matched label becomes attribution context;
- manifests retain sampler configuration and aggregate triage.

Builders accept only exact top-token matches for clean vanilla and promptless
graphs; prefix matches are not accepted. Suppressed-feature builders require a
corruption that changes every selected output. Loaders trust those generated
semantic relationships and validate only the row and graph field types.

Every builder emits the same `CircuitDatasetRow` shape with
`example_prompts`, `heldout_prompts`, and `control_prompts`. Visible tasks expose
five same-goal example prompts and continuations; the other tasksets leave
`example_prompts` empty and require policies to create arbitrary prompts through
the shared Circuit Tracer tools. Hidden target and control prompts retain maximal
graphs as an evaluation cache. Controls are selected during dataset generation,
so every dataset is self-contained and every loader is single-pass.

The original six training families remain available alongside nine sealed eval
families: two-hop recall, agreement, comparison, weekday arithmetic, path,
modular, rule, spatial, and Boolean composition. Every builder supports
`--split train`, `--split eval`, or (for cohort builders) `--split all`, plus an
explicit, repeatable `--family` override (`--families` is retained as an alias).

## TaskSets

### `circuit-supernodes`: visible-example discovery with causal holdouts

Each row contains visible evidence and two cached grading sets:

- Five same-goal prompt/continuation examples are shown without the semantic goal
  label. The policy must infer their shared concept.
- Set 1 is hidden: twenty-five additional same-goal positive holdouts. The
  submitted supernode and factor `-1` intervention must break every visible and
  hidden example.
- Set 2 is hidden: twenty-five different-goal neutral holdouts. The same
  intervention must preserve every clean target.

`circuit_attribute(prompt, feature_count)` attributes any prompt. In restoration
tasks its clean output/logit information is redacted, while
`circuit_intervene(source_prompt, node_id, prompt, intervention_weight)` maps a
source feature onto another arbitrary prompt and reports its baseline and
intervened top tokens. Stable `layer_feature` identities are remapped to their
highest-ranked prompt position in every graph. Reward is `1` only when the
visible-evidence, hidden-holdout, and neutral-control gates all pass. The goal
denotes the full semantic family rather than one answer label; a
spatial-composition circuit must therefore cover all eight directions.

Configs: `configs/{eval,rl}/visible-dataset.toml`.

### `supernode-rediscovery`: promptless goal supernodes

Each row contains a varied-answer positive cohort sharing one explicit semantic
goal key and a control cohort from different goals. Goal keys split meaningful
subfamilies such as analogy relations and antonym languages rather than grouping
unrelated prompts merely because they emit the same label.
Maximal Qwen graphs are stored for all prompts, but the policy receives neither
the subject prompts nor prompt, embedding, logit, or error nodes.
It uses the same arbitrary-prompt attribution and intervention tools as every
other taskset.

The final response contains named supernodes, their stable `layer_feature`
members, and one multiplicative factor per supernode (node-level overrides are
also accepted). For every positive prompt the verifier:

1. maps each stable feature to its highest-ranked stored prompt position;
2. negates each submitted multiplier;
3. applies those negative factors to the clean activations; and
4. requires the original top token to be displaced.

The same intervention is then applied to randomly sampled controls. Default
reward is binary: 100% of positive top tokens must change, while every control's
original top token must remain unchanged.

Configs: `configs/{eval,rl}/supernode-rediscovery.toml`.

### `suppressed-features`: corruption restoration

Generated prompts are answer-sampled normally. Within each same-family cohort,
the builder intersects stable features from the stored maximal clean graphs,
samples one shared 4–12-node signed multiplicative corruption with factors in
`[-8, 8]`, and retries until that corruption changes the output for every
hidden prompt. It then stores maximal
attribution graphs captured under the corruption.

The builder ensures hidden examples share stable layer/feature identities and
multiplicative factors (prompt positions may differ), and that every corrupted
graph changes its sampled clean output. Runtime loading intentionally does not
repeat those expensive generation-time semantic checks.

The policy creates its own prompts through the shared tools and submits stable
feature IDs with multiplicative repair factors. The hidden corruption factors
are never rendered to the policy. Final grading composes submitted factors
directly with each prompt's stored corruption, requires recovery of the clean top
token, and rewards mean clean-distribution similarity
`exp(-KL(clean || restored))`. The signed corruption can suppress, enhance, or
invert a feature activation.

Configs: `configs/{eval,rl}/suppressed-features.toml`.

## Modal service

| Purpose | App/class | Deploy command |
| --- | --- | --- |
| All sampling, attribution, intervention, corruption, and restoration | `circuit-supernodes/Attributor` | `uv run modal deploy modal_app/attribution.py` |

Deployments mutate external Modal state and are intentionally not part of local
unit tests.

## Dataset generation

Generated graph datasets and manifests are ignored by Git. Run from this
directory after deploying the relevant backend. Builders stream gzip-compressed
JSONL so full cohorts do not require tens of gigabytes of uncompressed local
storage or one equally large in-memory serialization.

All builders expose a typed Typer interface with documented defaults, accepted
split values, and numeric constraints:

```bash
uv run python scripts/build_visible_dataset.py --help
uv run python scripts/build_promptless_dataset.py --help
uv run python scripts/build_restoration_dataset.py --help
```

To select families directly instead of using a split, repeat `--family`:

```bash
uv run python scripts/build_visible_dataset.py \
  --family agreement --family comparison --out custom-cohorts.jsonl.gz
```

Visible-task hidden caches:

```bash
uv run python scripts/build_visible_dataset.py --n 2400 --tasks 16 --split train \
  --out circuit_supernodes/data/visible_dataset/train-cohorts.jsonl.gz
uv run python scripts/build_visible_dataset.py --n 2400 --tasks 16 --seed 1 --split eval \
  --out circuit_supernodes/data/visible_dataset/heldout-cohorts.jsonl.gz
```

Promptless cohorts:

```bash
uv run python scripts/build_promptless_dataset.py --split train \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes-train.jsonl.gz
uv run python scripts/build_promptless_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes.jsonl.gz
```

Suppressed-feature cohorts:

```bash
uv run python scripts/build_restoration_dataset.py --split train \
  --out circuit_supernodes/data/output_restoration/suppressed-features-train.jsonl.gz
uv run python scripts/build_restoration_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/output_restoration/suppressed-features.jsonl.gz
```

Each row stores only `family`, optional rediscovery metadata, and the three
cohorts. Each example stores only `prompt`, `target_token`, the consumed graph
fields, and optional corruption. The sidecar `*.manifest.json` contains the
content digest, dataset-schema version, family set, sampler configuration and
triage, cohort sizes, maximal-graph policy, and corruption provenance.

## Validation and execution

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest

uv run eval @ ../../configs/eval/visible-dataset.toml --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --no-push
uv run eval @ ../../configs/eval/suppressed-features.toml --no-push
```

The suppressed-feature egress canary is named
`suppressed-features-canary-egress.toml`.
