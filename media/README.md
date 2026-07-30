# Figure provenance

**`03_layer_order_composition.svg` and `04_separation_vs_layer_null.svg` are the
current figures.** Both regenerate from archived data with no manual step:

```bash
cd environments/circuit_supernodes
uv run python scripts/plot_layer_order.py
```

They are pure Python (no plotting library, no Node), and every number in them
traces to `archive/final/{opus5,sol56,kimik3}/traces.jsonl` and
`archive/oracle_layers_full.json` — the same data `FINDINGS.md` §4–5 quotes.
`scripts/plot_layer_order.py` reads those files directly; nothing here is
hand-copied.

## `03_layer_order_composition.svg` — where the reward's causal credit comes from

Twenty held-out graphs. Separation, split into what a layer ordering already
predicts with zero experiments (`separation_layer_null`) and what is left over
(`separation_over_null`), for the three frontier models run at xhigh reasoning
and for the seven scripted attacks from the oracle audit. Sorted by total
separation — the point is visible in the color split, not the ranking: the
tallest bars (models, layer-bands, community) are almost entirely the free
segment; the shortest (coherence-matched) is almost entirely the real one.

## `04_separation_vs_layer_null.svg` — models land on the same line as the attacks

The same twenty points plotted as separation against the layer-order null,
with a y = x reference line. On that line, separation equals what the layer
ordering already predicts — zero credit for anything else. All three models
sit close to it, indistinguishable in these terms from a partition built by
sorting features on layer number.

## What used to be here

`01_model_comparison.png` and `02_pass_at_k.png` are removed, not merely
flagged. Both were built from the v1 reward, whose only causal term
(`separation`) was flat across models — `reward = separation^0.6 ×
coverage^0.2 × completeness^0.2`, and grouping every displayed feature pins
`coverage` and `completeness` at 1.0 by construction — so the ranking they
showed came entirely from which model happened to group everything it was
shown, not from circuit quality:

```
$ uv run python scripts/reward_components.py archive/eval15/*/traces.jsonl
run          n  scored  separation  coverage  completeness  reward
fable       15      14       0.580     1.000         1.000   0.663
opus        15      11       0.558     0.701         0.321   0.276
sol         15      15       0.662     0.659         0.289   0.481
```

The model with the *worst* separation had the *best* reward. Neither PNG's
source data (48 rollouts for `01`, the pass@k sweep for `02`) was ever
committed, so neither could be regenerated or checked — the opposite of `03`
and `04`.

`02`'s claim ("sampling more rollouts helps only models that can already find
circuits") has no replacement here: it needs pass@k at k ≥ 3, and the archived
final runs are two rollouts per task, built to measure the layer-order
question, not resampling depth. That would be new data collection, not a
figure update, so it is left undone rather than faked.

## What is reproducible

- `environments/circuit_supernodes/archive/eval15/` — tracked `traces.jsonl`,
  `config.toml`, `eval.log` for the three models quoted above. Basis of the
  component table in this file and in `FINDINGS.md` §2.
  `scripts/reward_components.py` recovers the numbers from the traces.
- `environments/circuit_supernodes/archive/final/{opus5,sol56,kimik3}/` —
  trimmed traces (scores and metadata only) for the three xhigh runs behind
  `03` and `04`, and behind `FINDINGS.md` §5's headline table.
- `environments/circuit_supernodes/archive/oracle_layers_full.json` — the
  twenty-graph layer-bands vs. layer-spanning audit, and every other scripted
  attack's `separation_over_null`.
