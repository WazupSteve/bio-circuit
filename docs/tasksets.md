# TaskSets

The package exposes three task contracts over the same tool and dataset schema.
Visible discovery exposes an evidence cohort; promptless and restoration tasks
require the policy to create their own experimental prompts.

## Comparison

| TaskSet ID | Repository role | Subject | Hidden positives | Controls | Submission |
| --- | --- | --- | --- | --- | --- |
| `circuit-supernodes` | Shared-concept discovery from five visible examples | Qwen3-4B | 25 additional same-goal prompts | 25 different-goal prompts | Exactly one named supernode; omitted factors default to `1.0` |
| `supernode-rediscovery` | Promptless goal rediscovery | Qwen3-4B | Same goal with varied answers | Different-goal prompts | One or more named weighted supernodes |
| `suppressed-features` | Persistent-corruption restoration | Qwen3-4B | Same-family prompts sharing one hidden corruption | Not reward-bearing | Flat or grouped weighted nodes |

## Vanilla discovery: `circuit-supernodes`

The task prompt shows five prompt/continuation examples but does not reveal their
stored semantic goal. The policy must infer the shared concept, then use
attribution and transfer experiments to find one stable feature group necessary
for the full concept rather than one answer label.

The final response must contain exactly one `supernodes` group. Nodes may be bare
stable IDs because this adapter supplies the default factor `1.0`. The hidden
positive cache contains exactly 25 additional same-goal examples and the neutral
cache contains exactly 25 examples from other goals. With default configuration,
all five evidence examples and all hidden positives must change under the
negated proposal, while all controls remain unchanged.

Implementation: `tasksets/visible.py`. Configurations:
`configs/{eval,rl}/visible-dataset.toml`.

## Goal rediscovery: `supernode-rediscovery`

The policy receives only a semantic goal key, never the underlying cohort
prompts, answers, or stored graphs. Goal keys distinguish
mechanistically meaningful subfamilies rather than grouping every prompt that
happens to emit the same label.

The positive cohort is grouped by goal and is required to contain at least two
distinct full answers. Controls come from different semantic goals and may even
share an answer token with a positive. The policy must attach a factor to every node,
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
a shared 4–12-feature corruption with signed multipliers in `[-8, 8]`.
It keeps the corruption only if every selected prompt changes top token, then
stores the already-computed clean graphs for feature-coordinate resolution.

At runtime, arbitrary tool prompts receive ordinary clean feature attribution,
with all clean output logits, output tokens, and output edges redacted. The policy submits factors intended to identify and
reverse that mechanism. The accepted response can be a flat `nodes` list or
named `supernodes`; every node requires a factor.

All corrupted positives are stored in `heldout_prompts`; `example_prompts` is
empty because the policy creates its own experiments. Controls are clean prompts
from other families. This preserves the shared example/holdout/control dataset
shape while giving each field one consistent meaning.

Grading replays the hidden factors server-side, composes submitted multipliers
without negation, and rewards recovery of the clean distribution. The policy can observe feature structure and corrupted before/after experimental
outcomes, but receives neither clean output tokens nor hidden factor values. See [Graders](graders.md).

Implementation: `tasksets/restoration.py`. Configurations:
`configs/{eval,rl}/suppressed-features.toml`; a one-task external-model canary is
in `configs/eval/suppressed-features-canary-egress.toml`.

## Loading and discovery

Each adapter streams JSONL or gzip-compressed JSONL through strict Pydantic
validation and yields Verifiers tasks. Each public ID has an importable top-level package exporting exactly one TaskSet subclass through `__all__`; tests resolve all three with the Verifiers v1 loader. Default evaluation files download from the pinned public Hugging Face release
when absent locally. Custom or training files must be supplied explicitly.
