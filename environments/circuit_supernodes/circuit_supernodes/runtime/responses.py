"""Parse model responses and render model-facing graph text."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import ValidationError

from ..models import (
    Circuit,
    TracerGraphPayload,
    TracerLink,
)
from .graphs import (
    feature_nodes,
    influence_deltas,
    node_kind,
    normalized_layer,
    prompt_tokens,
    select_features,
    stable_feature_id,
    top_outputs,
)


def _validation_reason(exc: Exception) -> str:
    """Return the concise domain message from a validation failure."""

    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_input=False)
        if errors:
            error = max(errors, key=lambda item: len(item["loc"]))
            return error["msg"].removeprefix("Value error, ")
    return str(exc)


def circuit_from_response(
    response: str,
    *,
    default_weight: float | None = None,
    exact_groups: int | None = None,
    require_supernodes: bool = False,
) -> tuple[Circuit | None, str]:
    """Extract the final top-level JSON object and build a validated circuit."""

    try:
        decoder = json.JSONDecoder()
        parsed: list[tuple[int, dict[str, object]]] = []
        for match in re.finditer(r"\{", response):
            try:
                value, length = decoder.raw_decode(response[match.start() :])
            except Exception:
                continue
            if isinstance(value, dict):
                parsed.append((match.start() + length, value))
        if not parsed:
            raise ValueError("the final answer does not contain a JSON object")
        circuit = _normalize_circuit(
            max(parsed, key=lambda candidate: candidate[0])[1],
            default_weight=default_weight,
            exact_groups=exact_groups,
            require_supernodes=require_supernodes,
        )
        return circuit, ""
    except Exception as exc:
        return None, _validation_reason(exc)


def _normalize_circuit(
    value: dict[str, object],
    *,
    default_weight: float | None,
    exact_groups: int | None,
    require_supernodes: bool,
) -> Circuit:
    """Turn either accepted response shape into the canonical circuit model."""

    if ("nodes" in value) == ("supernodes" in value):
        raise ValueError("submit exactly one of nodes or supernodes")

    if "nodes" in value:
        if require_supernodes:
            raise ValueError("supernodes must be a non-empty list")
        groups = [{"name": "submitted", "nodes": value["nodes"]}]
    else:
        groups = cast(list[dict[str, Any]], value["supernodes"])

    if exact_groups is not None and len(groups) != exact_groups:
        raise ValueError(f"submit exactly {exact_groups} supernode")

    supernodes: dict[str, dict[str, float]] = {}
    for group in groups:
        name = cast(str, group["name"])
        if name in supernodes:
            raise ValueError("supernode names must be unique")
        group_weight = group.get("intervention_weight")
        weights: dict[str, float] = {}
        nodes = cast(list[str | dict[str, Any]], group["nodes"])
        for node in nodes:
            node_id = node if isinstance(node, str) else node["id"]
            node_weight = (
                None if isinstance(node, str) else node.get("intervention_weight")
            )
            feature_id = stable_feature_id(node_id)
            if feature_id in weights:
                raise ValueError("duplicate stable FEATURE id")
            weight = next(
                (
                    weight
                    for weight in (node_weight, group_weight)
                    if weight is not None
                ),
                default_weight,
            )
            if weight is None:
                raise ValueError("every node requires an intervention_weight")
            weights[feature_id] = cast(float, weight)
        supernodes[name] = weights

    return Circuit(supernodes=supernodes)


def render_graph(
    graph: TracerGraphPayload,
    max_edges_per_node: int = 5,
    *,
    include_labels: bool = True,
) -> str:
    """Render prompt tokens, top logits, nodes, and weighted links as text."""

    tokens = prompt_tokens(graph)
    outputs = top_outputs(graph)
    deltas = influence_deltas(graph)
    lines = [f"PROMPT TOKENS: {' | '.join(tokens)}"]
    if outputs:
        outputs = ", ".join(
            f"{token!r} (p={probability:.2f})" for token, probability in outputs[:5]
        )
        lines.append(f"TOP OUTPUTS: {outputs}")
    lines.append("")
    columns = "id | kind | layer | pos:token | activation | influence_delta"
    if include_labels:
        columns += " | label"
    lines.append(f"NODES ({columns}):")
    ordered = sorted(
        graph.nodes,
        key=lambda node: (
            node_kind(node) != "embedding",
            normalized_layer(node),
            node.ctx_idx,
        ),
    )
    for node in ordered:
        kind = node_kind(node)
        layer = normalized_layer(node)
        position = node.ctx_idx
        token = (
            tokens[position]
            if kind != "logit" and 0 <= position < len(tokens)
            else node.clerp
        )
        line = (
            f"  {node.node_id} | {kind} | L{layer} | "
            f"{position}:{token!r} | {(node.activation or 0.0):+.5g} "
            f"| {deltas[node.node_id]:.4f}"
        )
        if include_labels:
            line += f" | {(node.clerp or node.label)[:70]}"
        lines.append(line)
    lines.append("")
    lines.append("EDGES (source -> target : weight):")
    by_target: dict[str, list[TracerLink]] = {}
    for edge in graph.links:
        by_target.setdefault(edge.target, []).append(edge)
    for edges in (edges for _, edges in sorted(by_target.items())):
        edges.sort(key=lambda edge: -abs(edge.weight))
        lines.extend(
            f"  {edge.source} -> {edge.target} : {edge.weight:+.3f}"
            for edge in edges[:max_edges_per_node]
        )
    return "\n".join(lines)


def render_feature_view(
    graph: TracerGraphPayload,
    feature_count: int,
    *,
    include_labels: bool = True,
) -> dict[str, int | str]:
    """Render the top-influence-delta features with visible/available counts."""

    available = len(feature_nodes(graph))
    if not 1 <= feature_count <= available:
        raise ValueError(f"feature_count must be between 1 and {available}")
    view = select_features(graph, feature_count)
    return {
        "returned_feature_count": len(feature_nodes(view)),
        "available_feature_count": available,
        "graph": render_graph(view, include_labels=include_labels),
    }
