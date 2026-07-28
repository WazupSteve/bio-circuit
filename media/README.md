# Figure provenance

**Status: both charts in this directory are preliminary and must not be reused as
capability results.** They are kept as exhibits for the reward audit (STEPS.md §2),
not as findings.

## `01_model_comparison.png` — "Off-the-shelf models separate sharply"

GPT-5.6-sol 0.68 > Opus 4.8 0.64 > Fable 5 0.54 >> Qwen3.5-9B 0.14, against random
grouping 0.24 and hack 0.00; 16 eval examples × 48 rollouts per model.

Two problems, in order of severity:

1. **The v1 reward it plots is mostly not causal.** `reward = separation^0.6 ×
   coverage^0.2 × completeness^0.2`, and grouping every displayed feature makes
   coverage and completeness exactly 1.0 by construction (`rubric.py:204-205`,
   `rubric.py:249`). On the one archived run whose traces survive, the causal term is
   flat across models and the entire ranking comes from the two degenerate terms:

   ```
   $ uv run python scripts/reward_components.py archive/eval15/*/traces.jsonl
   run          n  scored  separation  coverage  completeness  reward
   fable       15      14       0.580     1.000         1.000   0.663
   opus        15      11       0.558     0.701         0.321   0.276
   sol         15      15       0.662     0.659         0.289   0.481
   ```

   The model with the *worst* separation has the *best* reward.

2. **It is not reproducible.** Only the PNG was ever committed — the 48-rollout
   traces, the eval config, and the rendering script were never tracked. Nothing in
   this repository can regenerate it or check its numbers.

Note that `archive/eval15` (15 tasks × 1 rollout, traces tracked) puts Fable *first*,
reversing this chart's ordering. Sample size differs, so this is not a contradiction
on its own — but combined with (1) there is no reason to trust either ordering as a
statement about circuit-finding ability.

The replacement is the v2 frontier baseline built in STEPS.md Stage 3. Until then,
any use of this figure must carry the component table above.

## `02_pass_at_k.png` — "Sampling more rollouts helps only models that can already
find circuits"

pass@k over 16 eval examples, 3 rollouts each, thresholded at reward ≥ 0.50. Inherits
problem (1) above in full — the threshold is applied to the same compromised reward,
so "can already find circuits" reads "already groups every feature". Also inherits
problem (2): source traces were never committed. k ≤ 3 is too small to say much
regardless.

## What *is* reproducible

`environments/circuit_supernodes/archive/eval15/` has tracked `traces.jsonl`,
`config.toml`, and `eval.log` for all three models. That run is the basis of the
component table above and of STEPS.md §2 finding 2. Its own summary chart
(`eval15_all_results.png`) was produced by a script that was never committed; the
numbers behind it are recoverable from the traces via `scripts/reward_components.py`.
