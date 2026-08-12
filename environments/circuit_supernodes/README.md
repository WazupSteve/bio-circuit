# Circuit task contracts

The package exports three independent Verifiers tasksets. Model-node-count is a
universal presentation policy, not a fourth plugin: each dataset stores the
maximal Circuit Tracer export (up to 4,096 feature nodes), and the policy chooses
an ordinary top-k view at inference time. Required, goal, corrupted, or held-out
nodes are never automatically inserted into those views.

## Shared answer sampling and families

Every generated task calls the same bounded greedy sampler before attribution:

- the subject is instructed to emit one answer from `possible_answers`;
- generation stops at the first complete answer label, within 50 tokens;
- correct matches are retained, wrong matches are counted and rejected, and
  exhaustion/errors are rejected;
- narration before the matched label becomes attribution context;
- rows and manifests retain sampler output, disposition, and target-token
  provenance.

The original six training families remain available alongside nine sealed eval
families: two-hop recall, agreement, comparison, weekday arithmetic, path,
modular, rule, spatial, and Boolean composition. Every builder supports
`--split train`, `--split eval`, or (for cohort builders) `--split all`, plus an
explicit `--families` override.

## Tasksets

### `circuit-supernodes`: vanilla causal MDL

Each row stores one maximal Qwen attribution graph and a response bank covering
the same full feature universe. `circuit_view_graph(feature_count)` chooses the
visible top-k subset. Ablation experiments and the final causal-MDL proposal are
validated against the maximal graph.

Configs: `configs/{eval,rl}/circuit-mdl.toml`.

### `supernode-rediscovery`: promptless goal supernodes

Each row contains a positive cohort whose sampled answers share one goal and a
random different-answer control cohort. Maximal Gemma graphs are stored for all
prompts, but the policy receives neither the subject prompts nor prompt tokens.
It calls `circuit_view_graph(cohort_index, feature_count)` to inspect label-free,
prompt-free top-k views of positive-cohort graphs.

The final response contains named supernodes, their stable `layer_feature`
members, and one multiplicative factor per supernode (node-level overrides are
also accepted). For every positive prompt the verifier:

1. maps each stable feature to its highest-ranked stored prompt position;
2. ablates the proposed set to establish that the answer depends on it;
3. applies the submitted multipliers; and
4. requires the desired answer token to be restored.

The same intervention is then applied to randomly sampled controls. Default
reward is binary: 100% of the positive cohort must shift from the ablated output
back to the desired answer, while 0% of controls may be driven to that answer.
The control gate rejects generic target-token activation circuits.

Configs: `configs/{eval,rl}/supernode-rediscovery.toml`.

### `suppressed-features`: cohort restoration

Generated prompts are answer-sampled normally. Within each same-family cohort,
the builder intersects stable features from the stored maximal clean graphs,
samples one shared 4–12-node multiplicative corruption, and retries until that
corruption changes the output for every discovery and held-out prompt. It then
stores maximal attribution graphs captured under the corruption.

The policy sees discovery prompts and desired tokens, and selects top-k views of
their corrupted graphs. It submits stable feature IDs and multiplicative repair
factors. Trials report discovery-cohort success only. Final scoring also replays
the proposal on hidden corrupted prompts. Reward is `1` only when restoration is
strictly greater than 80% on discovery prompts and strictly greater than 30% on
held-out prompts; otherwise it is `0`.

Configs: `configs/{eval,rl}/suppressed-features.toml`.

## Modal services

| Purpose | App/class | Deploy command |
| --- | --- | --- |
| Vanilla Qwen attribution/banks | `circuit-supernodes-attribution/Attributor` | `uv run modal deploy modal_app/attribution.py` |
| Promptless Gemma sampling/graphs/cohort verifier | `circuit-supernodes-gemma-attribution/GemmaAttributor` | `uv run modal deploy modal_app/attribution_gemma.py` |
| Qwen sampling/corruption/restoration | `circuit-supernodes-persistent-restoration/RestorationAttributor` | `uv run modal deploy modal_app/attribution_restoration.py` |

Deployments mutate external Modal state and are intentionally not part of local
unit tests.

## Dataset generation

Generated graph datasets and manifests are ignored by Git. Run from this
directory after deploying the relevant backend.

Vanilla MDL:

```bash
uv run python scripts/build_mdl_dataset.py --n 48 --split train \
  --out circuit_supernodes/data/circuit_mdl/train.jsonl
uv run python scripts/build_mdl_dataset.py --n 72 --seed 1 --split eval \
  --out circuit_supernodes/data/circuit_mdl/heldout.jsonl
uv run python scripts/precompute_response_banks.py \
  --input train.jsonl --out train-mdl.jsonl
uv run python scripts/precompute_response_banks.py \
  --input heldout.jsonl --out heldout-mdl.jsonl
```

Promptless cohorts:

```bash
uv run python scripts/build_promptless_dataset.py --split train \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes-train.jsonl
uv run python scripts/build_promptless_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/supernode_rediscovery/promptless-supernodes.jsonl
```

Suppressed-feature cohorts:

```bash
uv run python scripts/build_restoration_dataset.py --split train \
  --out circuit_supernodes/data/output_restoration/suppressed-features-train.jsonl
uv run python scripts/build_restoration_dataset.py --seed 1 --split eval \
  --out circuit_supernodes/data/output_restoration/suppressed-features.jsonl
```

Each builder writes a sidecar `*.manifest.json` containing the content digest,
family set, sampler triage, cohort sizes, graph policy, and corruption provenance.

## Validation and execution

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest

uv run eval @ ../../configs/eval/circuit-mdl.toml --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --no-push
uv run eval @ ../../configs/eval/suppressed-features.toml --no-push
```

The suppressed-feature egress and scoring-error canaries are named
`suppressed-features-canary-egress.toml` and
`suppressed-features-canary-error.toml`.
