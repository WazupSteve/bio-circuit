# Tasksets

Three tasksets share one tool set, one dataset row shape and the `circuit-null`
harness. They differ in what the policy sees and what the grader asks for.

| Taskset | Policy sees | Hidden grading data | Answer | Turns (eval) |
|---|---|---|---|---|
| `circuit-supernodes` | 5 example prompts and answers | 25 same-concept prompts, 25 controls | Exactly one supernode | 50 |
| `supernode-rediscovery` | A goal name | 4-prompt cohort, 4 controls | One or more weighted supernodes | 50 |
| `suppressed-features` | A family name | 10 corrupted prompts | Weighted nodes or supernodes | 50 |

Thresholds below are the class defaults (also used by the RL configs); the eval
configs relax some of them, as noted.

## circuit-supernodes

The prompt lists five prompt/continuation pairs without naming their shared
concept. The policy infers the concept, experiments, and submits one supernode
that is necessary for the whole concept rather than for one answer.

- Answer: exactly one entry in `supernodes`; node weights default to `1.0`.
- Graded on the 5 evidence examples, 25 hidden same-goal prompts and 25 controls
  from other goals.
- Defaults require every evidence and hidden prompt to change and every control
  to hold (`min_evidence_success = 1.0`, `min_holdout_success = 1.0`,
  `max_null_failure = 0.0`). The eval config uses 0.6 and 0.4 for the first two.

Source: `tasksets/visible.py`. Configs: `configs/{eval,rl}/visible-dataset.toml`.

## supernode-rediscovery

The policy gets only a goal name, which is the family name (e.g.
`boolean_composition`), and must write its own experiment prompts. It never sees
the cohort.

- Answer: one or more named supernodes; every node needs a weight, set on the
  node or on its group. Groups are merged into one intervention for grading.
- Graded on a 4-prompt cohort with at least two distinct answers and 4 controls
  from other goals (controls may share an answer token with the cohort). There
  is no second holdout set.
- Defaults: every cohort prompt changes, every control holds
  (`min_cohort_success = 1.0`, `max_control_activation = 0.0`). The eval config
  uses `min_cohort_success = 0.6`, i.e. 3 of 4.

Source: `tasksets/rediscovery.py`. Configs: `configs/{eval,rl}/supernode-rediscovery.toml`.

## suppressed-features

The subject carries a hidden, persistent corruption: 4–12 features shared by
the cohort, each scaled by a signed factor in [−8, 8], chosen so that every
cohort prompt changes its top token. The policy gets only the family name, sees
clean graphs with outputs redacted, and observes corrupted before/after top
tokens through `circuit_intervene`. It submits multipliers that undo the
corruption.

- Answer: a flat `nodes` list or named `supernodes`; every node needs a weight.
- Graded on the 10 corrupted prompts in `heldout_prompts` by how closely the
  restored output matches the clean distribution. `example_prompts` is empty;
  `control_prompts` exist in the data but are not scored.

Source: `tasksets/restoration.py`. Configs:
`configs/{eval,rl}/suppressed-features.toml`, plus a one-task canary on an
external model in `configs/eval/suppressed-features-canary-egress.toml`.

## Loading

Each taskset reads JSONL or gzipped JSONL. Files of 64 MB or more load lazily:
`load()` streams the file and keeps only task fields, and grading rereads just
the row it needs. Smaller files load fully. Without a local `dataset_file`, the
default evaluation file downloads from the pinned Hugging Face release; training
files must be supplied locally.

Each public ID is an importable package exporting one taskset class, resolved by
`verifiers.v1.utils.loaders.taskset_class`.
