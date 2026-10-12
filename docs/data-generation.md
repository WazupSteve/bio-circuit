# Data generation

Three Typer scripts in `environments/circuit_supernodes/scripts/` build the
datasets on a shared pipeline. Run them from `environments/circuit_supernodes`
with the subject backend deployed; they call it for sampling and attribution.

## Pipeline

```text
synthetic prompts (streamed in batches)
  -> greedy answer sampling (≤ 50 tokens) -> keep correct answers
  -> maximal clean attribution
  -> keep graphs whose top token equals the sampled answer and whose
     error-node share is ≤ --max-error-share (0.35)
  -> taskset-specific cohorts
  -> atomic JSONL(.gz) + manifest
```

**Prompt families** (`prompt_generation.py`). Training: antonym, analogy,
acronym, sequence, factual, incontext. Evaluation, requested explicitly:
two_hop, agreement, comparison, weekday, path_composition, modular_arithmetic,
rule_composition, spatial_composition, boolean_composition. Generation is
deterministic per seed, skips duplicate prompts, and retires small families that
run out of unique prompts. `--family` is repeatable and excludes a non-`all`
`--split`.

**Sampling.** Each prompt is prefixed with an instruction to answer with one
label. The subject samples greedily until the first complete allowed label; any
text before it becomes part of the attributed prompt. Wrong answers and failures
are counted in the manifest and dropped.

## circuit-supernodes

`build_visible_dataset.py` groups records by goal. Each row has 5 visible
examples, 25 hidden same-goal prompts and 25 controls from other goals; these
sizes are enforced. It caps records per goal to bound memory and can add
[Neuronpedia](neuronpedia.md) labels.

```bash
uv run python scripts/build_visible_dataset.py --n 2400 --tasks 16 --split train \
  --out circuit_supernodes/data/visible_dataset/train-cohorts.jsonl.gz
uv run python scripts/build_visible_dataset.py --n 2400 --tasks 16 --seed 1 --split eval \
  --out circuit_supernodes/data/visible_dataset/heldout-cohorts.jsonl.gz
```

## supernode-rediscovery

`build_promptless_dataset.py` groups by goal and needs at least two distinct
answers per cohort. Defaults: 4 cohort prompts, spread across answers where
possible, and 4 controls from other goals (which may share answer tokens).

```bash
uv run python scripts/build_promptless_dataset.py --split train \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes-train.jsonl.gz
uv run python scripts/build_promptless_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes.jsonl.gz
```

## suppressed-features

`build_restoration_dataset.py` groups clean graphs by family and target token,
then for each candidate cohort:

1. intersects the stable positive-activation features of all its prompts;
2. picks 4–12 of them with signed uniform factors in [−8, 8];
3. resolves each to its position on every prompt;
4. retries (up to `--attempts`) until the corruption changes every prompt's top token;
5. stores the clean graphs and the per-prompt corruption;
6. adds as many clean controls from other families.

Each row gets 10 corrupted prompts in `heldout_prompts` (`--heldout-size`) and an
empty `example_prompts`. Because cohorts are grouped by target token, all
prompts in a row share one answer.

```bash
uv run python scripts/build_restoration_dataset.py --split train \
  --out circuit_supernodes/data/output_restoration/suppressed-features-train.jsonl.gz
uv run python scripts/build_restoration_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/output_restoration/suppressed-features.jsonl.gz
```

## Output

Writers stream compact JSONL to a `.partial` file and rename it on success, so a
crash never leaves a truncated dataset. Every row is a `CircuitDatasetRow`. The
`*.manifest.json` sidecar records schema, SHA-256, row count, subject backend,
families, sampler triage, screening, generator settings, graph quality and
corruption provenance. Loaders validate types but trust the builders for
semantic properties.

Run each builder with `--help` for every option and default.
