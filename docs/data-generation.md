# Data generation

Three Typer scripts build the three TaskSet datasets over a shared pipeline. Run
them from `environments/circuit_supernodes` after deploying the corresponding
Modal backend.

## Common pipeline

```text
synthetic prompt generation
  -> bounded subject-model answer sampling
  -> keep correct answers only
  -> maximal clean attribution
  -> exact top-token faithfulness check
  -> graph-quality screen
  -> task-specific cohort construction
  -> atomic JSONL(.gz) + provenance manifest
```

### Prompt families

`scripts/prompt_generation.py` defines six training families:

- antonym, analogy, acronym, sequence, factual, and in-context lookup.

Nine sealed evaluation families must be requested explicitly:

- two-hop recall, subject-verb agreement, comparison, weekday arithmetic, path
  composition, modular arithmetic, rule composition, spatial composition, and
  Boolean composition.

Generation is deterministic for a seed and avoids duplicate prompt text. The
pool generator retires finite families after repeated collisions so a small
family does not cap the whole build. `--family` is repeatable; it is mutually
exclusive with an explicit non-`all` split.

### Answer preparation

Every prompt is prefixed with an instruction to emit only one answer label. The
subject model greedily samples up to 50 tokens and stops on the first complete
allowed label. Correct answers are retained; wrong matched answers and sampling
failures are counted in manifest triage but rejected. Any narration before the
answer is retained as part of the prompt attributed later.

### Graph screening

Accepted prompts receive maximal attribution graphs in bounded batches. A graph
is kept only when its top exported token exactly equals the sampled target token
and error-node influence is no greater than `--max-error-share` (default 0.35).
The pipeline records mean error influence and the share of feature influence
covered by the top 60 features.

## Vanilla/visible builder

`scripts/build_visible_dataset.py` groups accepted records by semantic goal. A
row contains five visible examples, 25 additional same-goal hidden positives,
and 25 randomly selected different-goal controls. The script enforces those
contract sizes even though it exposes size options. It limits retained records
per goal to bound memory and can optionally label top features through
Neuronpedia.

```bash
uv run python scripts/build_visible_dataset.py --n 2400 --tasks 16 --split train \
  --out circuit_supernodes/data/visible_dataset/train-cohorts.jsonl.gz
uv run python scripts/build_visible_dataset.py --n 2400 --tasks 16 --seed 1 \
  --split eval \
  --out circuit_supernodes/data/visible_dataset/heldout-cohorts.jsonl.gz
```

Backend: `circuit-supernodes/Attributor` (Qwen3-4B).

## Promptless rediscovery builder

`scripts/build_promptless_dataset.py` groups by semantic goal and requires at
least two distinct full answers in each positive cohort. Defaults select four
positives, preferentially spanning answer groups, and four controls from other
semantic goals. Controls may share output tokens with positives. Graph calls are
spawned concurrently in batches.

```bash
uv run python scripts/build_promptless_dataset.py --split train \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes-train.jsonl.gz
uv run python scripts/build_promptless_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes.jsonl.gz
```

Backend: `circuit-supernodes/Attributor` (Qwen3-4B).

## Suppressed-feature builder

`scripts/build_restoration_dataset.py` first groups faithful clean graphs by
family and target token. For each candidate cohort it:

1. intersects the top 2,048 positive-activation stable features across prompts;
2. selects 4–12 shared IDs and signed uniform factors in `[-8, 8]`;
3. resolves each ID to its prompt-specific position;
4. retries up to `--attempts` until every prompt changes top token;
5. retains the clean attribution graph, with output nodes and edges redacted; corruption is applied only during experiments and grading replay; and
6. adds equally many clean controls sampled from other families.

The default `heldout_size=10` produces one ten-item `heldout_prompts` list in the
stored row. `example_prompts` remains empty for this TaskSet.

```bash
uv run python scripts/build_restoration_dataset.py --split train \
  --out circuit_supernodes/data/output_restoration/suppressed-features-train.jsonl.gz
uv run python scripts/build_restoration_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/output_restoration/suppressed-features.jsonl.gz
```

Backend: `circuit-supernodes/Attributor` (Qwen3-4B).

## Artifact format and provenance

Writers stream compact JSONL, optionally gzip-compressed at low compression, to
a sibling `.partial` file and atomically rename it on success. This avoids both
large in-memory serialization and partially written final artifacts.

Every row uses `CircuitDatasetRow`. The checked-in fixture uses a supported legacy manifest shape. A sidecar `*.manifest.json` records the
dataset schema, SHA-256, row count, subject backend, families,
sampler triage, screening policy, maximal graph policy, generator parameters,
quality summaries, and corruption provenance where applicable. Manifests are provenance only and are not automatically checked. Runtime loaders
validate row and graph types but intentionally trust expensive semantic
relationships established by the builders.

Use `--help` on each builder for complete typed defaults and constraints.
