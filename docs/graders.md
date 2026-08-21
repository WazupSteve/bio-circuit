# Graders

All reward-bearing tasks use the same `AblationHoldoutGrader`. The restoration
module computes an additional diagnostic only. This separation is easy to miss
because the suppressed-feature task is named “restoration.”

## Response parsing

The grader scans the model's last reply for JSON objects and uses the one whose
parse ends latest. It accepts exactly one of these top-level shapes:

```json
{"nodes": [{"id": "12_345", "intervention_weight": 0.5}]}
```

```json
{"supernodes": [{"name": "mechanism", "nodes": ["12_345"], "intervention_weight": 1.0}]}
```

Task adapters decide whether a flat list is allowed, whether exactly one group
is required, and whether omitted weights receive a default. Validation rejects
duplicate feature IDs, duplicate case-insensitive group names, zero/non-finite
or out-of-range factors, more than eight groups, more than 60 total nodes, and
unknown fields. Invalid output receives reward zero without a remote replay.

## Shared causal grading algorithm

For every positive and control example:

1. Read the original top token from the stored graph. This, not the row's
   generator answer string, is the replay's expected token.
2. Flatten named groups into `(stable_feature_id, submitted_factor)` pairs.
3. Negate every factor.
4. Resolve each stable feature independently to its highest-influence occurrence
   in that example's graph. Missing features simply produce no coordinate.
5. Send the prompt, expected token, resolved interventions, and any stored
   baseline corruption to the appropriate Modal backend.
6. Rerun the model and compare baseline and intervened top tokens.

The backend validates that the expected token is top under the relevant
baseline. A positive succeeds when the intervention displaces it. A control
succeeds when the intervention preserves it. The response envelope and expected
token order are validated before rates are computed.

The factor used by reward is therefore:

```text
ordinary task:          changed activation = clean activation * (-submitted)
suppressed-feature task: changed activation = clean activation * corruption * (-submitted)
```

This sign convention means a submitted `1.0` is tested as sign inversion, not
zero ablation. It is intentional and versioned as
`v6-negated-intervention-ablation`.

## Gates and reward

The grader splits positives into the task adapter's “visible” prefix and its
remaining holdouts, then computes:

```text
visible_success_rate = changed visible positives / visible positives
holdout_success_rate = changed holdouts / holdouts
control_failure_rate = 1 - preserved controls / controls
reward = float(visible_pass AND holdout_pass AND controls_pass)
```

An empty visible partition has success rate `0`; an empty holdout partition
passes with rate `1`; empty controls pass with zero failure. Threshold names are
adapter-specific aliases over the same fields:

| Task | Positive gates | Specificity gate |
| --- | --- | --- |
| Vanilla | no generated visible examples; hidden same-goal holdout threshold | maximum different-goal null failure |
| Rediscovery | positive cohort is treated as the visible gate | maximum control activation/failure |
| Suppressed features | combined corrupted cohort is the holdout gate | maximum clean cross-family control failure |

Defaults are strict: required positive rate `1.0` and permitted control failure
`0.0`. Reward is binary, even though detailed rates are exposed as metrics.

## Restoration diagnostics

For suppressed-feature tasks only, `RestorationMetricEvaluator` replays the
stored corruption and composes the submitted factors without negating them:

```text
restored activation = clean activation * corruption * submitted
```

It compares the restored final-token distribution with the clean distribution.
If the clean target is not restored to top, similarity is zero. Otherwise:

```text
clean_similarity = exp(-KL(clean || restored))
```

The cohort metric is the arithmetic mean. Current task wiring calculates this
for `heldout_prompts`; `discovery_clean_similarity` is always reported as zero
because there are no `example_prompts`. These values never gate reward.

## Backend differences

Ordinary Qwen and Gemma services use the shared clean-ablation backend. Gemma
enforces a single, tokenizer-round-tripping expected token and validates the
activation tensor shape. The persistent Qwen backend supports a corrupted
baseline, coordinate-wise multiplier composition, NNSight intervention input
round-tripping, and separate full-distribution restoration replay.

Primary implementation: `grading/ablation.py`, `grading/restoration.py`,
`runtime/ablation_contract.py`, `runtime/modal_backend.py`, and
`modal_app/restoration_backend.py`.
