# Reference

## Feature IDs

| Form | Example | Used in |
|---|---|---|
| Positioned | `12_345_7` (layer, feature, token position) | Graph nodes, hidden corruptions |
| Stable | `12_345` (layer, feature) | Answers, `circuit_intervene` |

A stable ID is resolved separately on every prompt, to its highest-ranked
occurrence, so one feature can transfer across prompts and positions.

## Answer format

Named groups (all tasksets):

```json
{"supernodes": [{"name": "comparison",
                 "nodes": ["12_345", {"id": "18_9", "intervention_weight": 0.5}],
                 "intervention_weight": 1.0}]}
```

Flat list (restoration only):

```json
{"nodes": [{"id": "12_345", "intervention_weight": 0.5}]}
```

A node-level weight overrides its group's. Limits: 1–8 groups, at most 60
unique features, finite weights. The parser uses the last complete JSON object
in the final reply and ignores unrelated extra fields; the stored models reject
extras. Weight meaning differs by grader: see [graders](graders.md).

## Dataset rows

Every row has `family`, optional `goal`, and three lists: `example_prompts`,
`heldout_prompts`, `control_prompts`. Each example has `prompt`,
`target_token`, `graph` and optional `corruption` entries
`(layer, feature, position, factor)`.

| Taskset | Data folder | Eval file (public) | Train file (not released) |
|---|---|---|---|
| `circuit-supernodes` | `visible_dataset/` | `heldout-cohorts.jsonl.gz` | `train-cohorts.jsonl.gz` |
| `supernode-rediscovery` | `supernode_rediscovery/` | `promptless-supernodes.jsonl.gz` | `promptless-supernodes-train.jsonl.gz` |
| `suppressed-features` | `output_restoration/` | `suppressed-features.jsonl.gz` | `suppressed-features-train.jsonl.gz` |

Folders live under `environments/circuit_supernodes/circuit_supernodes/data/`
(ignored by Git). A local file takes precedence; otherwise the eval file
downloads from [amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit)
at revision `566f6880c28714a0e28cab3128535b6064b17a33`. Each file has 16 rows;
the Python wheel ships no data.

Each file has a `*.manifest.json` sidecar recording schema and generation-policy
versions, row count, SHA-256, subject model, transcoders, pruning thresholds,
sampler and generator settings, screening and graph quality. Manifests are
provenance only; loaders don't check them.

## Config keys

| Key | Meaning |
|---|---|
| `env.taskset.id` | Taskset ID |
| `env.taskset.dataset_file` | Dataset file name in the taskset's data folder |
| `env.taskset.min_*` / `max_*` | Grading thresholds ([tasksets](tasksets.md)) |
| `env.taskset.subject.*` | Subject model, transcoders, pruning |
| `env.agent.max_turns` | Turn limit |
| `env.agent.harness.id` / `tool_timeout` | `circuit-null`; 3600 s for long tool calls |

Changing subject or threshold values changes results; compare runs only with
matching settings.

## Backend RPCs

Internal methods on the Modal `Attributor` and the HTTP server:
`sample_until_answer`, `attribute_prompt`, `evaluate_ablation_cohort`,
`test_corruption_coordinates`, `restore_output`. The policy-facing tools are
the stable interface.

## What "hidden" means

Hidden data is kept out of policy prompts and tool output. Operators, graders,
backend code and stored task data can all read it; nothing is encrypted, and the
public dataset contains every hidden cohort. Keep raw data files away from the
policy for a meaningful evaluation.
