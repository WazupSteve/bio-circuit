# Graders

The two discovery tasks share a negative-intervention causal grader. The
restoration task has a separate reward that directly composes the submitted
factors with a hidden corruption and measures KL divergence from clean behavior.

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
duplicate feature IDs, duplicate case-insensitive group names, non-finite factors,
more than eight groups, and more than 60 total nodes. Incoming normalization tolerates
unrelated extra fields; canonical models reject extras. Invalid output receives reward zero without a remote replay.

## Discovery ablation grading

Vanilla visible discovery and promptless rediscovery use
`AblationHoldoutGrader`. For every positive and control example it:

1. Reads the original top token from the stored graph.
2. Flattens named groups into `(stable_feature_id, submitted_factor)` pairs.
3. Negates every factor.
4. Resolves each stable feature to its highest-influence occurrence in that
   example's graph.
5. Reruns the prompt and compares baseline and intervened top tokens.

A positive succeeds when the original top token is displaced. A control
succeeds when it remains top. This tests whether a discovered mechanism is both
necessary and specific. The discovery factor is:

```text
changed activation = clean activation * (-submitted factor)
```

This sign convention is part of the discovery grading contract.

The visible task separately grades its five shown evidence examples and 25
hidden same-concept examples. Promptless treats its hidden positive cohort as
the first gate. Both also enforce a control-failure ceiling. Default thresholds
require every positive to change and every control to remain unchanged, yielding
a binary reward.

## Restoration KL grading

Suppressed-feature restoration does not use the ablation grader. Every dataset
example contains a coordinate-level corruption, but those factors remain hidden
in server-side task data. They are not rendered in the task prompt, graph view,
or tool response.

For clean activation `a`, hidden corruption `c`, and submitted restoration `r`,
the backend executes:

```text
restored activation = a * c * r
```

The submitted factor is applied directly and is never negated. For a feature
that is truly corrupted, its stable ID is mapped to the hidden prompt-specific
coordinate server-side. Other submitted features use ordinary highest-ranked
position resolution.

The backend computes `KL(clean || restored)` over the complete final-token
distribution. An example's score is zero unless the original clean target token
is restored to top-1; after that gate its score is:

```text
similarity = exp(-KL(clean || restored))
```

Final reward is the arithmetic mean of this gated similarity over the hidden
corrupted cohort. Thus reward is continuous in `[0, 1]`, reaches one only at the
clean distribution.
Reported metrics include mean raw KL, mean similarity, and restored-target rate.

Clean cross-family controls remain in the universal dataset row for artifact
compatibility but do not participate in restoration reward; clean-distribution
KL on the corrupted target cohort is the grading objective.

## Backend validation

Restoration replay requires the clean target to be top before corruption and the
stored corruption to change it. Builder-generated corruption multipliers lie in
`[-8, 8]`; policy proposals may use any finite multiplier, including zero to
clamp an activation. Backend
protocol mismatches raise errors rather than silently becoming low rewards.

Primary implementation: `grading/ablation.py`, `grading/restoration.py`,
`runtime/ablation_contract.py`, `runtime/corruption.py`, and
`modal_app/backend/`.
