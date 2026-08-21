# Core environment

The core is a Verifiers multi-turn environment in which a policy model can
attribute arbitrary text, inspect a bounded view of its attribution graph, apply
multiplicative feature interventions, and finally submit a circuit. The actual
subject-model computation runs in a Modal GPU service; the local package provides
typed requests, graph manipulation, tool rendering, and remote clients.

## End-to-end flow

```text
arbitrary prompt
  -> optional bounded answer sampling
  -> subject-model forward pass
  -> Circuit Tracer attribution (up to 4,096 feature nodes)
  -> exported graph JSON
  -> top-k textual view for the policy model
  -> stable feature selection (`layer_feature`)
  -> resolve feature to highest-ranked position on a target prompt
  -> multiply its clean activation by the requested factor
  -> re-run model and compare baseline/intervened top tokens
```

The sampling phase is primarily a dataset-generation operation. During a task,
the policy-facing tools expose attribution and intervention, not a separate raw
sampling tool.

## Subject-model sampling

`runtime/sampling.py` implements greedy generation for at most 50 new tokens.
The request supplies a prompt and a finite set of answer labels. After each
token, the sampler searches for the first complete, case-insensitive answer
label at word boundaries. It returns:

- `matched`, with generated text, matched answer, attribution-ready prompt, and
  the first answer token;
- `exhausted`, when no label appears before EOS or the token bound; or
- `error`, for invalid input or model failure.

Text generated before the answer label is appended to the original prompt and
becomes attribution context. Dataset preparation keeps only samples whose
matched answer equals the generator's expected answer. A later faithfulness
check requires the attributed graph's top exported token to equal the sampled
target token exactly.

## Attribution graphs

The Modal backend loads a Circuit Tracer `ReplacementModel` and calls
`circuit_tracer.attribute`. Ordinary exports use:

- up to 4,096 feature nodes;
- up to five output logits and a desired cumulative logit probability of 0.95;
- node threshold 0.8 and edge threshold 0.98 when serializing.

The repository consumes a deliberately small graph schema: prompt tokens, nodes,
and weighted links. Nodes may be feature, embedding, error, or logit nodes.
Feature importance is reconstructed as the nonnegative increment between
successive cumulative influence values. That increment determines ranking.

Stored datasets retain maximal exports. `attribute(prompt, feature_count)` then
selects an ordinary top-k view at request time, retaining embeddings and logits
and only edges whose endpoints remain. Required, corrupted, or submitted nodes
are not automatically inserted into normal policy views.

The rendered response includes prompt tokens, top outputs, feature coordinates,
activations, influence deltas, labels, and at most five strongest incoming edges
per target. It also reports returned and available feature counts.

## Stable identity and position resolution

Circuit Tracer feature nodes use `layer_feature_position`. Policy submissions use
`layer_feature`, because token position changes between prompts. For every graph,
the runtime ranks feature occurrences by influence delta and maps a stable ID to
its highest-ranked occurrence. An ID absent from a target graph produces no
intervention for that prompt.

This resolution happens independently for every source, experiment, and grading
graph. It is what makes cross-prompt feature transfer possible, but also means a
proposal is effective only on prompts where the stable feature was attributed.

## Multiplicative patching

For a clean activation `a` and submitted factor `w`, ordinary intervention writes
the absolute activation `a * w` through Circuit Tracer's feature-intervention
API, then reruns the model. Policy factors may be any finite float; zero is a
valid multiplicative clamp. The `[-8, 8]` bound applies only when generating
hidden dataset corruptions.

The suppressed-feature backend additionally has a hidden signed corruption
factor `c`.
A restoration replay composes factors coordinate-wise:

```text
executed activation = a * c * w
```

Factors on coordinates not present in the other set implicitly multiply by one.
The restoration reward path uses this direct composition without negating the
policy factor; see [Graders](graders.md). Hidden corruption factors stay in task
data and backend calls and are never included in policy-facing prompts or tool
responses.

## Policy-facing tools

All three TaskSets install `CircuitExperimentTools`:

- `circuit_attribute(prompt, feature_count)` attributes any prompt. For a
  suppressed-feature task, the task's persistent corruption is applied before
  attribution, so the returned graph represents corrupted execution.
- `circuit_intervene(source_prompt, node_id, prompt, intervention_weight)` checks
  that the stable node exists in the source graph, resolves it independently on
  the target graph, reruns the target with the multiplier, and returns baseline
  and intervened top-token fields.

Tool errors are returned as concise `{ "error": ... }` objects instead of
terminating a rollout. Modal calls retry up to four attempts with waits of 1, 4,
and 15 seconds.

## Schemas and limits

`models.py` is the main validation boundary. A final `Circuit` contains one to
eight named supernodes, at most 60 unique stable features total, and one bounded
factor per feature. Unknown fields are rejected. Dataset rows share one
shape across all tasks: `family`, optional `goal`, plus
`example_prompts`, `heldout_prompts`, and `control_prompts`. Each example stores
the prompt, target token, graph, and optional coordinate-level corruption.

Relevant source modules are `runtime/circuit_tracer.py`, `runtime/graphs.py`,
`runtime/responses.py`, `runtime/modal_backend.py`, `runtime/corruption.py`, and
`models.py` under `environments/circuit_supernodes/circuit_supernodes`.

### Run-configured subject backend

Training and evaluation tasksets select the subject model and graph pruning in
their own TOML configuration. The shared Modal app and `Attributor` class remain
global; these values parameterize its container instance:

```toml
[taskset.subject]
model_name = "Qwen/Qwen3-4B"
transcoder_set = "mwhanna/qwen3-4b-transcoders"
node_threshold = 0.8
edge_threshold = 0.98
```

RL configurations use the corresponding `[env.taskset.subject]` and
`[eval.env.taskset.subject]` tables. Model and transcoder changes produce a
distinct parameterized Modal container, while tools and graders within the run
share the same subject configuration.
