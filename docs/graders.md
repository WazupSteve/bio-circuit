# Graders

The two discovery tasksets share an ablation grader. Restoration has its own
grader that measures distance from the clean output.

## Parsing the answer

The parser reads the policy's last reply, takes the last complete JSON object,
and accepts exactly one of:

```json
{"supernodes": [{"name": "mechanism", "nodes": ["12_345"], "intervention_weight": 1.0}]}
```

```json
{"nodes": [{"id": "12_345", "intervention_weight": 0.5}]}
```

Each taskset decides whether the flat form is allowed, how many groups are
required, and whether missing weights get a default. The parser rejects
duplicate features, duplicate group names (case-insensitive), non-finite
weights, more than eight groups and more than 60 features. An invalid answer
scores 0 without contacting the backend.

## Ablation grading (discovery)

`AblationHoldoutGrader` (`grading/ablation.py`) handles `circuit-supernodes`
and `supernode-rediscovery`. For every graded prompt it:

1. reads the expected top token from the stored graph;
2. merges groups into `(feature, weight)` pairs;
3. **negates each weight**;
4. resolves each feature to its highest-ranked position in that prompt's graph;
5. reruns the prompt and compares top tokens.

```text
changed activation = clean activation × (−weight)
```

A positive prompt succeeds when its expected token is displaced (to any other
token). A control succeeds when its expected token stays on top. The task
passes, with reward 1, only if every gate meets its threshold: positive success
rates at or above `min_*_success` and control failure at or below the control
threshold. Otherwise reward is 0.

`circuit-supernodes` grades its 5 evidence examples and 25 hidden prompts as
separate gates; `supernode-rediscovery` grades its cohort as the only positive
gate. Defaults are in [tasksets](tasksets.md).

`circuit_intervene` applies weights without negation; the rediscovery prompt
tells the policy this.

## Restoration grading

`RestorationMetricEvaluator` (`grading/restoration.py`) replays each of the 10
corrupted prompts with the submitted factors composed onto the hidden
corruption:

```text
restored activation = clean activation × corruption × weight
```

Weights are not negated. A submitted feature that is part of the corruption
maps to its hidden coordinate on each prompt; other features use the
highest-ranked position.

Each prompt scores 0 unless the clean top token is back on top; otherwise:

```text
similarity = exp(−KL(clean ‖ restored))
```

over the full final-token distribution. Reward is the mean similarity, in
[0, 1], reaching 1 only at the clean distribution. Metrics also report mean KL
and the restored-target rate.

## Replay validation

Before scoring, every replay must reproduce its stored baseline: the expected
token must still be top and, for restoration, the corruption must still change
it. A failure on any prompt marks the whole task `baseline_drift` or
`grading_validation_error` with reward 0, recorded in the trace metrics.

Source: `grading/`, `runtime/ablation_contract.py`, `runtime/corruption.py`,
`modal_app/backend/interventions.py`.
