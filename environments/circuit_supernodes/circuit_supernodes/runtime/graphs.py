"""Circuit, coordinate, and Circuit Tracer graph operations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from ..models import (
    Circuit,
    CoordinateTuple,
    Intervention,
    TracerGraphPayload,
    TracerNode,
)

# Bound attribution work before influence-based pruning. Maximal exports and
# every top-k view share this graph contract.
MAX_FEATURE_NODES = 2048

NodeKind = Literal["feature", "embedding", "error", "logit"]


def stable_feature_id(node_id: str) -> str:
    """Normalize ``layer_feature[_position]`` to ``layer_feature``."""

    match = re.fullmatch(r"(\d+)_(\d+)(?:_(\d+))?", node_id.strip())
    if match is None:
        if node_id.strip().upper().startswith("E_"):
            raise ValueError(f"{node_id!r} is an embedding node, not a FEATURE")
        raise ValueError(f"invalid FEATURE id {node_id!r}")
    return f"{int(match.group(1))}_{int(match.group(2))}"


def parse_coordinate(node_id: str) -> CoordinateTuple:
    """Parse a Circuit Tracer ``layer_feature_position`` node ID."""

    match = re.fullmatch(r"(\d+)_(\d+)_(\d+)", node_id)
    if match is None:
        raise ValueError(f"invalid FEATURE id {node_id!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def weighted_nodes(circuit: Circuit) -> list[tuple[str, float]]:
    """Flatten a circuit's named groups in declaration order."""

    return [
        (node_id, weight)
        for supernode in circuit.supernodes.values()
        for node_id, weight in supernode.items()
    ]


def node_kind(node: TracerNode) -> NodeKind:
    """Normalize Circuit Tracer's exporter feature type."""

    feature_type = node.feature_type.lower()
    return (
        "embedding"
        if "embed" in feature_type
        else "error"
        if "error" in feature_type
        else "logit"
        if "logit" in feature_type
        else "feature"
    )


def normalized_layer(node: TracerNode) -> int:
    """Return a numeric layer for sorting exporter nodes."""

    try:
        return int(node.layer)
    except Exception:
        return 0


def prompt_tokens(graph: TracerGraphPayload) -> list[str]:
    return graph.metadata.prompt_tokens


def feature_nodes(graph: TracerGraphPayload) -> list[TracerNode]:
    return [node for node in graph.nodes if node_kind(node) == "feature"]


def node_by_id(graph: TracerGraphPayload) -> dict[str, TracerNode]:
    return {node.node_id: node for node in graph.nodes}


def influence_deltas(graph: TracerGraphPayload) -> dict[str, float]:
    """Derive per-node increments from cumulative influence without mutation."""

    previous = 0.0
    deltas: dict[str, float] = {}
    for node in sorted(graph.nodes, key=lambda item: item.influence or 0.0):
        influence = node.influence or 0.0
        deltas[node.node_id] = max(influence - previous, 0.0)
        previous = influence
    return deltas


def ranked_features(graph: TracerGraphPayload) -> list[TracerNode]:
    deltas = influence_deltas(graph)
    return sorted(feature_nodes(graph), key=lambda node: -deltas[node.node_id])


def top_outputs(graph: TracerGraphPayload) -> list[tuple[str, float]]:
    """Return logit labels ordered by exported token probability."""

    outputs = []
    for node in graph.nodes:
        if node_kind(node) != "logit":
            continue
        match = re.search(r'"(.*?)"', node.clerp, re.DOTALL)
        outputs.append((match.group(1) if match else node.clerp, node.token_prob))
    return sorted(outputs, key=lambda item: -item[1])


def select_features(
    graph: TracerGraphPayload,
    limit: int = 60,
    required_feature_ids: set[str] | None = None,
) -> TracerGraphPayload:
    """Return a top-feature graph view while retaining embeddings and logits."""

    required = required_feature_ids or set()
    if len(required) > limit:
        raise ValueError("required features exceed the visible feature budget")
    features = feature_nodes(graph)
    feature_ids = {node.node_id for node in features}
    if missing := required - feature_ids:
        raise ValueError(
            f"required features are absent from the graph: {sorted(missing)}"
        )
    ranked = ranked_features(graph)
    selected = [node for node in ranked if node.node_id in required]
    selected.extend(node for node in ranked if node.node_id not in required)
    keep = {node.node_id for node in selected[:limit]} | {
        node.node_id
        for node in graph.nodes
        if node_kind(node) in ("embedding", "logit")
    }
    return TracerGraphPayload(
        metadata=graph.metadata.model_copy(deep=True),
        nodes=[
            node.model_copy(deep=True) for node in graph.nodes if node.node_id in keep
        ],
        links=[
            edge.model_copy(deep=True)
            for edge in graph.links
            if edge.source in keep and edge.target in keep
        ],
    )


def non_feature_kinds(
    graph: TracerGraphPayload, stable_ids: Sequence[str]
) -> dict[str, NodeKind]:
    """Map requested ids that name only logit or error nodes to that kind.

    Logit and error node ids share the ``layer_index_position`` shape of
    FEATURE ids, so they parse but never resolve to an intervention.
    """

    requested = set(stable_ids)
    features = {stable_feature_id(node.node_id) for node in feature_nodes(graph)}
    kinds: dict[str, NodeKind] = {}
    for node in graph.nodes:
        kind = node_kind(node)
        if kind not in ("logit", "error"):
            continue
        try:
            stable_id = stable_feature_id(node.node_id)
        except ValueError:
            continue
        if stable_id in requested and stable_id not in features:
            kinds.setdefault(stable_id, kind)
    return kinds


def resolve_interventions(
    graph: TracerGraphPayload, interventions: Sequence[tuple[str, float]]
) -> list[Intervention]:
    """Resolve stable feature IDs to their highest-delta graph positions."""

    weights = dict(interventions)
    selected: dict[str, TracerNode] = {}
    for node in ranked_features(graph):
        try:
            stable_id = stable_feature_id(node.node_id)
        except Exception:
            continue
        if stable_id in weights and stable_id not in selected:
            selected[stable_id] = node
    resolved = [
        (*parse_coordinate(node.node_id), weights[stable_id])
        for stable_id, node in selected.items()
    ]
    return sorted(resolved)


def graph_quality(
    graph: TracerGraphPayload, max_features: int = 60
) -> dict[str, float | int]:
    """Return attribution fractions and top-feature coverage."""

    deltas = influence_deltas(graph)
    total = sum(deltas.values())
    if total <= 0:
        return {"error": 1.0, "feature": 0.0, "displayed": 0.0, "features": 0}
    kinds: dict[str, float] = {}
    for node in graph.nodes:
        kind = node_kind(node)
        kinds[kind] = kinds.get(kind, 0.0) + deltas[node.node_id]
    features = ranked_features(graph)
    feature_total = sum(deltas[node.node_id] for node in features)
    return {
        "error": kinds.get("error", 0.0) / total,
        "feature": kinds.get("feature", 0.0) / total,
        "displayed": (
            sum(deltas[node.node_id] for node in features[:max_features])
            / feature_total
            if feature_total > 0
            else 0.0
        ),
        "features": len(features),
    }
