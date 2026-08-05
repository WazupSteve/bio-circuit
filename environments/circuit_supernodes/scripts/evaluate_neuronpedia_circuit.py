"""Evaluate a curated Neuronpedia circuit against the fixed MDL reward.

The script downloads a Neuronpedia attribution graph, preserves its curated
supernodes, constructs a feature universe containing those nodes plus the most
important remaining nodes, and builds (or reuses) a Modal causal-response bank.
It compares the curated grouping with two size- and topology-matched controls:

* regrouped: the same curated nodes randomly reassigned to the same group sizes;
* random_features: random nodes from the same fixed feature universe.

The null hypothesis is the zero-response predictor represented by
``MDLScore.null_error``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from itertools import combinations
from pathlib import Path
from statistics import fmean, median
from typing import Any

from circuit_supernodes.graphs import feature_coords, load_graph_json
from circuit_supernodes.mdl import OUTPUT, ResponseBank, score_circuit
from circuit_supernodes.rubric import MODAL_APP, MODAL_CLS, validate_topology

DEFAULT_URL = (
    "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/"
    "user-graphs/anonymous/qwen-name.json"
)


def download_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "biocircuit-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def curated_supernodes(data: dict[str, Any]) -> dict[str, list[str]]:
    raw_groups = data.get("qParams", {}).get("supernodes", [])
    groups = {str(group[0]): [str(node) for node in group[1:]] for group in raw_groups}
    if not groups or any(len(members) < 2 for members in groups.values()):
        raise ValueError("Neuronpedia graph has no usable curated supernodes")
    return groups


def feature_universe(
    data: dict[str, Any], groups: dict[str, list[str]], max_features: int
) -> tuple[list[str], dict[str, int]]:
    graph = load_graph_json(data)
    curated = {node for members in groups.values() for node in members}
    missing = sorted(curated - set(graph.nodes))
    if missing:
        raise ValueError(f"curated nodes missing from graph: {missing}")
    if len(curated) > max_features:
        raise ValueError("max_features is smaller than the curated circuit")

    ranked = sorted(graph.feature_nodes, key=lambda node: -node.share)
    ranks = {node.id: index + 1 for index, node in enumerate(ranked)}
    fillers = [node.id for node in ranked if node.id not in curated]
    selected = sorted(curated | set(fillers[: max_features - len(curated)]))
    return selected, ranks


def graph_derived_edges(
    data: dict[str, Any], groups: dict[str, list[str]]
) -> list[list[str]]:
    """Project the graph to a deterministic acyclic, transitively reduced DAG.

    An inter-group link is allowed only from the group with the lower median
    layer to the group with the higher median layer. Every group with a direct
    target-logit link initially reaches OUTPUT. Redundant direct OUTPUT edges
    are then removed. This rule is fixed before looking at the response bank.
    """

    graph = load_graph_json(data)
    membership = {
        node_id: name for name, members in groups.items() for node_id in members
    }
    median_layer = {
        name: median(graph.nodes[node_id].layer for node_id in members)
        for name, members in groups.items()
    }
    output_ids = {node.id for node in graph.nodes.values() if node.kind == "logit"}
    observed: set[tuple[str, str]] = set()
    for edge in graph.edges:
        source = membership.get(edge.source)
        target = membership.get(edge.target)
        if source is not None and target is not None and source != target:
            if median_layer[source] < median_layer[target]:
                observed.add((source, target))
        elif source is not None and edge.target in output_ids:
            observed.add((source, OUTPUT))

    # Remove a direct output edge when another outgoing group edge already gives
    # the source a path to a group that reaches OUTPUT.
    group_edges = {(source, target) for source, target in observed if target != OUTPUT}
    output_sources = {source for source, target in observed if target == OUTPUT}

    def reaches_output(start: str, without_direct: bool = False) -> bool:
        seen: set[str] = set()
        stack = [target for source, target in group_edges if source == start]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in output_sources:
                return True
            stack.extend(target for source, target in group_edges if source == current)
        return start in output_sources and not without_direct

    minimal_outputs = {
        source
        for source in output_sources
        if not reaches_output(source, without_direct=True)
    }
    edges = sorted(group_edges | {(source, OUTPUT) for source in minimal_outputs})
    closure, error = validate_topology(list(groups), [list(edge) for edge in edges])
    if closure is None:
        raise ValueError(f"projected Neuronpedia topology is invalid: {error}")
    return [list(edge) for edge in edges]


def score(
    bank: ResponseBank,
    groups: dict[str, list[str]],
    edges: list[list[str]],
) -> dict[str, float | int]:
    closure, error = validate_topology(list(groups), edges)
    if closure is None:
        raise ValueError(error)
    return score_circuit(
        bank,
        groups,
        closure,
        edge_count=len(edges),
    ).to_dict()


def summarize(
    values: list[float], observed: float, *, exact: bool = False
) -> dict[str, float | int]:
    ordered = sorted(values)
    upper_count = sum(value >= observed for value in values)
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "p05": ordered[int(0.05 * (len(ordered) - 1))],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "observed_percentile": sum(value <= observed for value in values) / len(values),
        "upper_tail_p": (
            upper_count / len(values)
            if exact
            else (1 + upper_count) / (len(values) + 1)
        ),
    }


def exact_assignments(
    nodes: tuple[str, ...], names: list[str], sizes: list[int]
) -> list[dict[str, list[str]]]:
    """Enumerate every labeled partition with the requested group sizes."""

    if len(names) == 1:
        return [{names[0]: list(nodes)}]
    assignments: list[dict[str, list[str]]] = []
    first_name, first_size = names[0], sizes[0]
    for selected in combinations(nodes, first_size):
        selected_set = set(selected)
        remainder = tuple(node for node in nodes if node not in selected_set)
        for suffix in exact_assignments(remainder, names[1:], sizes[1:]):
            assignments.append({first_name: list(selected), **suffix})
    return assignments


def load_or_build_bank(
    path: Path,
    prompt: str,
    feature_ids: list[str],
    target_token: str,
    modal_app: str,
    modal_class: str,
) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    import modal

    attributor = modal.Cls.from_name(modal_app, modal_class)()
    bank = attributor.causal_response_bank.remote(
        prompt,
        feature_coords(feature_ids),
        target_token,
        [0.25, 0.5],
        [0.75, 1.0],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, separators=(",", ":")) + "\n")
    return bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--max-features", type=int, default=60)
    parser.add_argument("--controls", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modal-app", default=MODAL_APP)
    parser.add_argument("--modal-class", default=MODAL_CLS)
    args = parser.parse_args()
    if args.controls < 1:
        raise SystemExit("--controls must be positive")

    data = download_json(args.url)
    graph = load_graph_json(data)
    groups = curated_supernodes(data)
    feature_ids, ranks = feature_universe(data, groups, args.max_features)
    edges = graph_derived_edges(data, groups)
    prompt = str(data["metadata"]["prompt"])
    target_token = graph.top_outputs[0][0]
    bank_data = load_or_build_bank(
        args.bank_cache,
        prompt,
        feature_ids,
        target_token,
        args.modal_app,
        args.modal_class,
    )
    bank = ResponseBank.from_dict(bank_data)
    if list(bank.feature_ids) != feature_ids:
        raise ValueError("cached response-bank universe does not match this audit")

    observed = score(bank, groups, edges)
    subcircuits: list[dict[str, Any]] = []
    for group_count in range(1, len(groups) + 1):
        for selected_names in combinations(groups, group_count):
            selected_groups = {name: groups[name] for name in selected_names}
            selected_edges = graph_derived_edges(data, selected_groups)
            subcircuits.append(
                {
                    "groups": list(selected_names),
                    "derived_edges": selected_edges,
                    "score": score(bank, selected_groups, selected_edges),
                }
            )
    rng = random.Random(args.seed)
    names = list(groups)
    sizes = [len(groups[name]) for name in names]
    curated_ids = [node for name in names for node in groups[name]]

    def assign(nodes: list[str]) -> dict[str, list[str]]:
        assigned: dict[str, list[str]] = {}
        offset = 0
        for name, size in zip(names, sizes, strict=True):
            assigned[name] = nodes[offset : offset + size]
            offset += size
        return assigned

    partition_count = math.factorial(len(curated_ids)) // math.prod(
        math.factorial(size) for size in sizes
    )
    exact_regrouping = partition_count <= 100_000
    if exact_regrouping:
        regrouped_scores = [
            score(bank, assignment, edges)
            for assignment in exact_assignments(tuple(curated_ids), names, sizes)
        ]
    else:
        regrouped_scores = []
        for _ in range(args.controls):
            shuffled = rng.sample(curated_ids, len(curated_ids))
            regrouped_scores.append(score(bank, assign(shuffled), edges))
    random_feature_scores: list[dict[str, float | int]] = []

    for _ in range(args.controls):
        sampled = rng.sample(feature_ids, len(curated_ids))
        random_feature_scores.append(score(bank, assign(sampled), edges))

    best_subcircuit = max(
        subcircuits, key=lambda item: float(item["score"]["net_bits"])
    )
    best_names = list(best_subcircuit["groups"])
    best_sizes = [len(groups[name]) for name in best_names]
    best_edges = list(best_subcircuit["derived_edges"])
    best_control_rng = random.Random(args.seed + 1)
    best_random_scores: list[dict[str, float | int]] = []
    for _ in range(args.controls):
        sampled = best_control_rng.sample(feature_ids, sum(best_sizes))
        assigned: dict[str, list[str]] = {}
        offset = 0
        for name, size in zip(best_names, best_sizes, strict=True):
            assigned[name] = sampled[offset : offset + size]
            offset += size
        best_random_scores.append(score(bank, assigned, best_edges))

    best_same_pool_scores: list[dict[str, float | int]] = []
    if len(best_names) == 1:
        best_name = best_names[0]
        for members in combinations(curated_ids, best_sizes[0]):
            best_same_pool_scores.append(
                score(bank, {best_name: list(members)}, best_edges)
            )

    metric = "net_bits"
    result = {
        "source": {
            "url": args.url,
            "slug": data.get("metadata", {}).get("slug"),
            "model": data.get("metadata", {}).get("scan"),
            "transcoder_urls": data.get("metadata", {})
            .get("info", {})
            .get("source_urls", []),
            "prompt": prompt,
            "target_token": target_token,
        },
        "audit": {
            "seed": args.seed,
            "controls": args.controls,
            "feature_universe_size": len(feature_ids),
            "calibration_matrices": len(bank.calibration),
            "validation_matrices": len(bank.validation),
            "bank_id": bank.metadata.get("bank_id"),
        },
        "circuit": {
            "supernodes": groups,
            "curated_node_importance_ranks": {
                node: ranks[node] for node in curated_ids
            },
            "derived_edges": edges,
            "score": observed,
        },
        "curated_subcircuits": subcircuits,
        "controls": {
            "regrouped_same_nodes": {
                "exact": exact_regrouping,
                "partition_population": partition_count,
                **summarize(
                    [float(item[metric]) for item in regrouped_scores],
                    float(observed[metric]),
                    exact=exact_regrouping,
                ),
            },
            "random_features_same_size": summarize(
                [float(item[metric]) for item in random_feature_scores],
                float(observed[metric]),
            ),
            "best_subcircuit_random_features": {
                "groups": best_names,
                **summarize(
                    [float(item[metric]) for item in best_random_scores],
                    float(best_subcircuit["score"][metric]),
                ),
            },
            "best_subcircuit_same_curated_pool": (
                {
                    "groups": best_names,
                    "exact": True,
                    **summarize(
                        [float(item[metric]) for item in best_same_pool_scores],
                        float(best_subcircuit["score"][metric]),
                        exact=True,
                    ),
                }
                if best_same_pool_scores
                else None
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
