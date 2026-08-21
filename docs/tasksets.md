# TaskSets

The package exposes three task contracts over the same tool and dataset schema.
In all current generated rows, `example_prompts` is empty: the policy creates its
own experimental prompts, while stored hidden cohorts are reserved for grading.

## Comparison

| TaskSet ID | Repository role | Subject | Hidden positives | Controls | Submission |
| --- | --- | --- | --- | --- | --- |
| `circuit-supernodes` | Vanilla/visible arbitrary-prompt discovery | Qwen3-4B | 25 same-goal prompts | 25 different-goal prompts | Exactly one named supernode; omitted factors default to `1.0` |
| `supernode-rediscovery` | Promptless goal rediscovery | Gemma-2-2B | Same goal, answer, and first token | Different answer and first token | One or more named weighted supernodes |
| `suppressed-features` | Persistent-corruption discovery/restoration | Qwen3-4B | Same-family prompts sharing one corruption | Clean cross-family prompts | Flat or grouped weighted nodes |

## Vanilla discovery: `circuit-supernodes`

The task prompt reveals a semantic goal, but no example prompt. The policy uses
arbitrary attribution and transfer experiments to find one stable feature group
necessary for the full goal rather than one answer label. This distinction is
especially important for families such as spatial composition, where the same
mechanism produces multiple direction labels.

The final response must contain exactly one `supernodes` group. Nodes may be bare
stable IDs because this adapter supplies the default factor `1.0`. The hidden
positive cache contains exactly 25 same-goal examples and the neutral cache
contains exactly 25 examples from other goals. With default configuration, all
positive top tokens must change under the negated proposal and all control top
tokens must remain unchanged.

Implementation: `tasksets/visible.py`. Configurations:
`configs/{eval,rl}/visible-dataset.toml`.

## Goal rediscovery: `supernode-rediscovery`

The policy receives a semantic goal key and desired answer, but never receives
the underlying cohort prompts or their stored graphs. Goal keys distinguish
mechanistically meaningful subfamilies rather than grouping every prompt that
happens to emit the same label.

The positive cohort is grouped by goal, full normalized answer, and attributed
first token. Controls are chosen only when both full answer and first token
differ from the desired outcome. The policy must attach a factor to every node,
either on the enclosing supernode or on individual nodes. Multiple named groups
are accepted, but grading flattens them into one intervention proposal.

The adapter treats the entire positive cohort as its first grading gate and has
no second holdout partition. Defaults require every positive output to change
and every control output to be preserved.

Implementation: `tasksets/rediscovery.py`. Configurations:
`configs/{eval,rl}/supernode-rediscovery.toml`.

## Suppressed-feature restoration: `suppressed-features`

Each task represents a semantic family with a persistent hidden feature
corruption. The builder first finds clean same-family prompts with the same
target token, intersects their stable positive-activation features, and samples
a shared 4–12-feature corruption with positive multipliers in `[2, 8]`. It keeps
the corruption only if every selected prompt changes top token, then attributes
and stores the corrupted executions.

At runtime, arbitrary tool prompts are attributed under the same stable
corruption specification. The policy submits factors intended to identify and
reverse that mechanism. The accepted response can be a flat `nodes` list or
named `supernodes`; every node requires a factor.

All stored corrupted examples are currently placed in `heldout_prompts` even
though builder CLI names describe two contributions (`cohort_size` and
`heldout_size`). Consequently the task's visible/evidence success rate is empty
and the reward-bearing positive gate is the combined hidden cohort. Controls are
clean prompts from other families.

The task also reports clean-distribution restoration similarity, but that metric
does not affect reward. See [Graders](graders.md) for the important distinction
between restoration replay and the negative-ablation reward contract.

Implementation: `tasksets/restoration.py`. Configurations:
`configs/{eval,rl}/suppressed-features.toml`; a one-task external-model canary is
in `configs/eval/suppressed-features-canary-egress.toml`.

## Loading and discovery

Each adapter streams JSONL or gzip-compressed JSONL through strict Pydantic
validation and yields Verifiers tasks. The root package lazily exports the
vanilla `CircuitTaskset`; the other TaskSet IDs rely on the runner/plugin
discovery conventions. Missing data files fail with the builder command needed
to create them.
