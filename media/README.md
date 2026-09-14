# Rollout analysis

This folder summarizes the final 96-rollout comparison: three policy models,
16 tasks, and two rollouts per task. PNG files are sized for direct use in a
blog post; matching SVG files are included for lossless editing.

## Headline results

| Model | Successes | Gradeable success | Reported API cost | Cost / success | Median rollout | Avg. tool calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 | 6/32 | 6/22 (27.3%) | $11.37 | $1.89 | 29.8 min | 19.3 |
| Fable 5.1 | 7/32 | 7/22 (31.8%) | $123.43 | $17.63 | 22.0 min | 17.5 |
| GPT-6 Astra | 5/32 | 5/22 (22.7%) | $55.50 | $11.10 | 37.9 min | 50.4 |

- Fable recorded the most successes, but Kimi was much cheaper per success.
- Eleven of 18 successes came from the two two-hop tasks. Weekday task 8
  contributed five more and spatial-composition task 6 contributed two.
- Task 11 was the only unanimous result: every model succeeded on both
  rollouts.
- Null-control specificity was Kimi's main bottleneck: 13/22 gradeable
  rollouts passed the null gate, versus 19/22 for Fable and 20/22 for Astra.
- Astra made 50.4 tool calls per rollout on average, about 2.6 times Kimi's
  19.3, without converting the extra experiments into a higher success rate.
- Kimi and Fable placed 85% and 82% of their submitted features in layers
  28–35. Astra placed 44% there and searched earlier layers more broadly.

## Important caveat

Tasks 4, 7, 9, 10, and 12 showed `baseline_drift` for both rollouts under every
model. Those 30 records completed normally, but scoring marked their subject
baseline invalid. The gradeable rates therefore use 22 rollouts per model,
while raw rates use all 32. With only two rollouts per task/model, task-level
differences should be treated as descriptive rather than statistically stable.

The reported cost is the sum of `usage.cost` in policy-model API calls. It does
not include Modal GPU costs for the Qwen3-4B subject model.

## Figures

1. `01_outcome_overview` — success, gradeable failure, and baseline drift.
2. `02_task_model_matrix` — exact 0/2, 1/2, or 2/2 result for every task/model.
3. `03_family_performance` — success rate by reasoning family.
4. `04_scoring_gates` — evidence, holdout, null-control, and final pass rates.
5. `05_tool_strategy` — average attribution, graph-query, and intervention calls.
6. `06_cost_efficiency` — reported spend, cost per success, and median duration.
7. `07_feature_layer_selection` — where submitted `layer_feature` nodes landed.
8. `08_rollout_efficiency` — tool calls versus wall time for all 96 rollouts.

The underlying tables are in `rollouts.csv`, `model_summary.csv`, and
`task_model_matrix.csv`.

## View rollouts

Run this from the repository root:

```bash
cd /home/amitprakash/bio-circuit
uv run prime eval view \
  --outputs-dir media/viewer \
  --env-dir environments \
  --limit 10
```

The installed Prime CLI 0.6.29 viewer only discovers the legacy
`outputs/evals/.../metadata.json + results.jsonl` layout. The completed runs use
the newer `config.toml + traces.jsonl` layout, so pointing the viewer directly
at `outputs/` finds nothing. `media/viewer/` is a viewer-compatible transcript
export of the three final runs.

Regenerate every chart, CSV, and viewer export with:

```bash
uv run scripts/analyze_rollouts.py
```
