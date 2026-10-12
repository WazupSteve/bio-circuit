# Neuronpedia labels

Optional enrichment for dataset generation; tasks and graders never call
Neuronpedia. Only the visible builder uses it, behind `--neuronpedia-labels`.

For each graph, `scripts/neuronpedia.py` takes the top 60 feature nodes, reduces
them to unique `(layer, feature)` pairs and sends one batch request to
`https://www.neuronpedia.org/api/features` (model `qwen3-4b`, layer
`<layer>-transcoder-hp`). The first non-empty explanation per feature is written
to the `label` of every matching node.

Failures are logged and generation continues without labels. Requests time out
after 30 seconds, with no authentication, caching, retries or rate limiting.

Graph views show a node's `clerp` text first, so a label appears only when
`clerp` is empty. Labels never affect ranking, interventions, cohorts or
grading.
