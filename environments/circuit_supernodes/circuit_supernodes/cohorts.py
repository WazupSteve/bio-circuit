"""Shared schemas and scoring helpers for intervention cohorts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .graphs import Graph, GraphNode
from .interventions import MAX_ABS_WEIGHT, MAX_INTERVENTIONS

_FEATURE_ID = re.compile(r"^(\d+)_(\d+)(?:_(\d+))?$")


def stable_feature_id(node_id: str) -> str | None:
    """Normalize ``layer_feature[_position]`` to ``layer_feature``."""

    match = _FEATURE_ID.fullmatch(str(node_id).strip())
    if match is None:
        return None
    return f"{int(match.group(1))}_{int(match.group(2))}"


def parse_json_object(text: str | None) -> dict[str, Any] | None:
    """Return the last JSON object in a response, allowing fenced output."""

    if not text:
        return None
    decoder = json.JSONDecoder()
    parsed: list[tuple[int, int, dict[str, Any]]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, length = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append((match.start(), match.start() + length, value))
    top_level = [
        candidate
        for candidate in parsed
        if not any(
            outer_start < candidate[0] and candidate[1] <= outer_end
            for outer_start, outer_end, _ in parsed
        )
    ]
    return top_level[-1][2] if top_level else None


def _weighted_node(raw: object, *, label: str) -> tuple[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    node_id = str(raw.get("id", "")).strip()
    stable = stable_feature_id(node_id)
    if stable is None:
        raise ValueError(f"{label} has invalid FEATURE id {node_id!r}")
    raw_weight = raw.get("intervention_weight")
    if isinstance(raw_weight, bool):
        raise ValueError(f"{label} needs a numeric intervention_weight")
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} needs a numeric intervention_weight") from exc
    if not math.isfinite(weight) or abs(weight) > MAX_ABS_WEIGHT:
        raise ValueError(
            f"{label} intervention_weight must be finite and in "
            f"[-{MAX_ABS_WEIGHT:g}, {MAX_ABS_WEIGHT:g}]"
        )
    return stable, weight


def parse_weighted_nodes(
    text: str | None, *, require_supernodes: bool = False
) -> tuple[list[tuple[str, float]] | None, str]:
    """Parse weighted nodes, flattening named supernodes when requested.

    Restoration uses ``{"nodes": [{...}]}``. Promptless discovery uses
    ``{"supernodes": [{"name": ..., "nodes": [{...}]}]}`` so the policy has
    to state its semantic grouping as well as the intervention.
    """

    data = parse_json_object(text)
    if data is None:
        return None, "the final answer does not contain a JSON object"

    raw_nodes: list[object] = []
    if require_supernodes:
        raw_groups = data.get("supernodes")
        if not isinstance(raw_groups, list) or not raw_groups:
            return None, "supernodes must be a non-empty list"
        names: set[str] = set()
        for index, raw_group in enumerate(raw_groups):
            if not isinstance(raw_group, Mapping):
                return None, f"supernode {index} must be an object"
            name = str(raw_group.get("name", "")).strip().casefold()
            if not name or name in names:
                return None, "supernode names must be non-empty and unique"
            names.add(name)
            nodes = raw_group.get("nodes")
            if not isinstance(nodes, list) or not nodes:
                return None, f"supernode {index} needs a non-empty node list"
            group_weight = raw_group.get("intervention_weight")
            for node in nodes:
                if isinstance(node, str):
                    raw_nodes.append({"id": node, "intervention_weight": group_weight})
                elif isinstance(node, Mapping) and "intervention_weight" not in node:
                    raw_nodes.append({**node, "intervention_weight": group_weight})
                else:
                    raw_nodes.append(node)
    else:
        nodes = data.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            return None, "nodes must be a non-empty list"
        raw_nodes = list(nodes)

    if len(raw_nodes) > MAX_INTERVENTIONS:
        return None, f"use at most {MAX_INTERVENTIONS} intervention nodes"
    parsed: list[tuple[str, float]] = []
    seen: set[str] = set()
    try:
        for index, raw_node in enumerate(raw_nodes):
            item = _weighted_node(raw_node, label=f"node {index}")
            if item[0] in seen:
                raise ValueError(f"duplicate stable FEATURE id {item[0]!r}")
            seen.add(item[0])
            parsed.append(item)
    except ValueError as exc:
        return None, str(exc)
    return parsed, ""


def expand_stable_interventions(
    graph: Graph, interventions: Sequence[tuple[str, float]]
) -> list[tuple[int, int, int, float]]:
    """Apply each stable feature at its highest-ranked exported prompt position."""

    weights = dict(interventions)
    selected: dict[str, GraphNode] = {}
    for node in sorted(graph.feature_nodes, key=lambda item: -item.share):
        key = stable_feature_id(node.id)
        if key in weights and key not in selected:
            selected[key] = node
    expanded = []
    for key, node in selected.items():
        layer, feature, position = map(int, node.id.split("_"))
        expanded.append((layer, feature, position, weights[key]))
    return sorted(expanded)


def validate_against_graphs(
    graphs: Sequence[Graph], interventions: Sequence[tuple[str, float]]
) -> tuple[bool, str]:
    """Require every proposed stable feature to occur in a visible cohort graph."""

    exported = {
        stable
        for graph in graphs
        for node in graph.feature_nodes
        if (stable := stable_feature_id(node.id)) is not None
    }
    missing = sorted({node_id for node_id, _ in interventions} - exported)
    if missing:
        return False, f"FEATURE ids are absent from the exported cohort: {missing}"
    return True, ""


def success_rate(results: Sequence[Mapping[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(bool(result.get(key)) for result in results) / len(results)


def threshold_score(
    discovery_rate: float,
    heldout_rate: float,
    *,
    discovery_threshold: float,
    heldout_threshold: float,
) -> dict[str, float | bool]:
    """Strict cohort gates requested by the task contract."""

    discovery_pass = discovery_rate > discovery_threshold
    heldout_pass = heldout_rate > heldout_threshold
    return {
        "discovery_success_rate": discovery_rate,
        "heldout_success_rate": heldout_rate,
        "discovery_threshold": discovery_threshold,
        "heldout_threshold": heldout_threshold,
        "discovery_pass": discovery_pass,
        "heldout_pass": heldout_pass,
        "reward": float(discovery_pass and heldout_pass),
    }
