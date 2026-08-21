# Neuronpedia backing

Neuronpedia is optional enrichment, not a runtime dependency of the tasks or
graders. It is currently wired only into the vanilla visible-dataset builder via
`--neuronpedia-labels`.

For each graph, `scripts/neuronpedia.py` selects the ordinary top 60 feature
nodes, reduces positional IDs to unique `(layer, feature)` pairs, and sends one
batch POST request to:

```text
https://www.neuronpedia.org/api/features
```

Requests identify model `qwen3-4b`, layer `<layer>-transcoder-hp`, feature index,
and ask for one maximum-activation record. The helper takes the first non-empty
explanation description returned for each feature and writes it into every
matching positional node's `label` field in the in-memory graph.

Failures are deliberately non-fatal in the builder: they are logged and dataset
generation continues without labels. The request has a 30-second timeout. There
is no authentication, response cache, retry layer, rate-limit handling, or
Gemma mapping in this repository.

At runtime, graph rendering displays `clerp` text before the optional label, so a
Neuronpedia label is visible only when the exporter does not already provide a
non-empty `clerp` for that node. Labels have no effect on feature ranking,
intervention resolution, cohort construction, or grading.

This is the entirety of the current Neuronpedia backing; there is no local
Neuronpedia database or proxy service.
