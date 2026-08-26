"""Optional Neuronpedia enrichment for generated attribution graphs."""

import httpx
from circuit_supernodes.models import TracerGraphPayload
from circuit_supernodes.runtime.graphs import feature_nodes, select_features

NEURONPEDIA_FEATURES_URL = "https://www.neuronpedia.org/api/features"


def add_feature_labels(graph: TracerGraphPayload, max_features: int = 60) -> int:
    """Add available Neuronpedia descriptions to the graph's top features."""

    nodes = feature_nodes(select_features(graph, max_features))
    features = {tuple(map(int, node.node_id.split("_")[:2])) for node in nodes}
    if not features:
        return 0
    payload = [
        {
            "modelId": "qwen3-4b",
            "layer": f"{layer}-transcoder-hp",
            "index": feature,
            "maxActsToReturn": 1,
        }
        for layer, feature in sorted(features)
    ]
    response = httpx.post(NEURONPEDIA_FEATURES_URL, json=payload, timeout=30)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list):
        raise ValueError("unexpected Neuronpedia response")

    labels = {}
    for record in records:
        descriptions = record.get("explanations") or []
        label = next(
            (
                str(item.get("description", "")).strip()
                for item in descriptions
                if str(item.get("description", "")).strip()
            ),
            "",
        )
        if label:
            layer = str(record["layer"]).split("-", 1)[0]
            labels[f"{layer}_{record['index']}"] = label

    count = 0
    for node in feature_nodes(graph):
        if label := labels.get("_".join(node.node_id.split("_")[:2])):
            node.label = label
            count += 1
    return count
