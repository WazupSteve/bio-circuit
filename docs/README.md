# Bio-circuit system documentation

This directory documents the repository as it exists today. The shortest useful
mental model is:

```text
policy model
  -> CircuitExperimentTools
     -> Modal subject model (sample, attribute, intervene)
  -> final circuit JSON
     -> causal ablation grader (discovery) or clean-KL replay (restoration)
```

Dataset builders use the same subject-model services ahead of time to generate
prompts, retain correct model answers, export maximal attribution graphs, and
assemble the hidden cohorts consumed by the TaskSets.

## Reading order

1. [Core environment](core-environment.md) — model sampling, Circuit Tracer
   graphs, stable feature identities, multiplicative patching, and tool calls.
2. [TaskSets](tasksets.md) — the vanilla/visible discovery, promptless
   rediscovery, and suppressed-feature restoration contracts.
3. [Graders](graders.md) — response parsing, discovery ablation gates, and
   clean-distribution restoration reward.
4. [Data generation](data-generation.md) — prompt families, answer-aware
   sampling, graph screening, cohort construction, artifacts, and commands.
5. [Neuronpedia](neuronpedia.md) — optional feature-label enrichment.
6. [Repository map and feature-creep inventory](repository-map.md) — deployment,
   configuration, packaging, tests, and systems outside the requested core.

## Important terminology

- **Subject model**: the remotely hosted model whose internal features are
  attributed and patched (Qwen3-4B or Gemma-2-2B, depending on the task).
- **Policy model**: the model solving a Verifiers task by calling tools and
  submitting circuit JSON.
- **Feature node**: a transcoder feature at a layer and prompt position, exported
  as `layer_feature_position`.
- **Stable feature ID**: `layer_feature`. It deliberately omits position so the
  same learned feature can be transferred between prompts.
- **Multiplier**: any finite factor selected by the policy and applied to a
  feature's clean activation; zero clamps it to zero.
- **Corruption**: a stored positive multiplier applied before a restoration
  proposal in the suppressed-feature task.
- **Ablation**: the reward grader negates the submitted multiplier and checks
  whether the original top token is displaced.

## Source boundaries

The root project supplies Prime/Verifiers runner configuration. The nested
`environments/circuit_supernodes` project is the independently buildable TaskSet
package and owns runtime code, builders, Modal services, and tests. Generated
datasets are mostly ignored by Git; the checked-in visible heldout artifact is an
exception.
