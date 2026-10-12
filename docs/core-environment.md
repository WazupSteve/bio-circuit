# Core environment

A policy model experiments on Qwen3-4B through three tools, then submits a
circuit as JSON. The subject model runs in a Modal GPU service (or a
[self-hosted server](local.md)); the Python package provides typed requests,
graph handling, the tools and the graders.

## Experiment loop

```text
policy prompt
  -> circuit_attribute: Circuit Tracer attribution (up to 2,048 feature nodes)
  -> circuit_query_graph: bounded pages of nodes, edges, tokens
  -> choose a stable feature (`layer_feature`)
  -> circuit_intervene: resolve it on another prompt, scale its activation,
     rerun, compare top tokens
  -> final answer: JSON circuit, graded on hidden prompts
```

## Tools

All three tasksets install `CircuitExperimentTools` (`runtime/circuit_tracer.py`).
Tool errors come back as `{"error": "..."}` instead of ending the rollout.

**`circuit_attribute(prompt)`** attributes the prompt, stores the graph in a
temporary SQLite file and returns a graph ID, counts and a short output summary.
Only the latest graph is kept: a new attribution deletes the previous one, and
old IDs return "Graph has been replaced".

**`circuit_query_graph(graph_id, view, offset, limit, node_id, kind)`** reads one
page of the stored graph:

- `view="nodes"`: nodes ranked by influence; filter by `node_id` (positioned or
  stable ID) and `kind` (`feature`, `embedding`, `error`, `logit`).
- `view="edges"`: edges ranked by absolute weight; `node_id` matches either end.
- `view="tokens"`: prompt tokens by position.

Pages hold at most 50 items and 12,000 bytes, and text is cut to 300 characters
(`text_truncated` is set). Follow `next_offset` with the same filters; `null`
marks the end.

**`circuit_intervene(source_prompt, node_id, prompt, intervention_weight)`**
checks that the stable feature exists in the source prompt's graph, resolves it
on the target prompt, multiplies its activation by `intervention_weight`, reruns
the target and reports `baseline_top_token`, `intervened_top_token` and
`changed_top_token` (discovery tasks add expected-token fields). It uses
temporary graphs and does not replace the stored one.

Modal calls retry up to four times, waiting 1, 4 and 15 seconds.

## Attribution graphs

The backend calls `circuit_tracer.attribute` with at most 2,048 feature nodes,
five output logits and a 0.95 cumulative logit probability, then prunes at node
threshold 0.8 and edge threshold 0.98. Graphs keep prompt tokens, nodes
(feature, embedding, error, logit) and weighted links. A feature's influence is
the non-negative increase in cumulative influence it adds, which also sets its
rank.

## Stable features and positions

Graph nodes are `layer_feature_position` (`12_345_7`); answers use the stable
`layer_feature` (`12_345`) because positions differ between prompts. On every
graph the runtime maps a stable ID to its highest-ranked occurrence and
intervenes there only. An ID absent from a graph has no effect on that prompt,
so a circuit can only act where its features were attributed.

## Interventions

For clean activation `a` and factor `w`, an intervention writes `a × w` and
reruns the model. `w` can be any finite number: `0` zero-ablates, negative
values flip the sign.

The graders interpret factors differently per task:

- Discovery grading negates every submitted factor (`a × −w`); see
  [graders](graders.md).
- Restoration composes the submitted factor with the hidden corruption `c`:
  `a × c × w`.

## Restoration mode

When a task carries a hidden corruption (`suppressed-features`):

- `circuit_attribute` returns the clean graph with logit nodes, their edges and
  top outputs removed.
- `circuit_intervene` applies the hidden corruption features that appear in the
  experiment prompt's graph, then reports only top tokens. A prompt with none of
  them runs as the clean model would.

Corruption features and factors never appear in prompts or tool output.

## Subject configuration

Each config selects the subject model and pruning; tools and graders in a run
share it:

```toml
[env.taskset.subject]
model_name = "Qwen/Qwen3-4B"
transcoder_set = "mwhanna/qwen3-4b-transcoders"
node_threshold = 0.8
edge_threshold = 0.98
```

RL configs use the same table under `[[env]]` and `[[eval.env]]`. Each distinct
model/transcoder pair gets its own parameterized Modal container.

## Source

`runtime/circuit_tracer.py` (tools), `runtime/graph_store.py` (stored graph and
pages), `runtime/graphs.py` (ranking, resolution), `runtime/modal.py` (Modal and
HTTP clients), `runtime/responses.py` (answer parsing), `runtime/corruption.py`,
`models.py`, and `modal_app/backend/` (GPU side).
