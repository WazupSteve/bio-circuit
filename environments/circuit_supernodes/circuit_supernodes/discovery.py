"""Deterministic supernode discovery from a causal response bank.

This module deliberately has no model, prompt, label, or network dependency.
It groups features whose incoming and outgoing calibration responses match,
induces a sparse DAG, and selects circuit size by leave-one-calibration-condition-
out MDL. The response bank's held-out validation matrices are read only after a
proposal has been selected, to report its final score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .mdl import MAX_GROUPS, OUTPUT, MDLScore, ResponseBank, score_circuit


@dataclass(frozen=True)
class _Candidate:
    groups: dict[str, tuple[str, ...]]
    edges: tuple[tuple[str, str], ...]
    closure: dict[str, set[str]]
    selection_net_bits: float


class _CalibrationObjective:
    """Cross-validated objective containing calibration matrices only."""

    def __init__(self, bank: ResponseBank) -> None:
        matrices = bank.calibration
        self.mode = (
            "leave-one-calibration-condition-out"
            if len(matrices) > 1
            else "calibration-resubstitution"
        )
        folds: list[ResponseBank] = []
        for held_out in range(len(matrices)):
            fitted = tuple(
                matrix for index, matrix in enumerate(matrices) if index != held_out
            )
            if not fitted:
                fitted = (matrices[held_out],)
            folds.append(
                ResponseBank(
                    feature_ids=bank.feature_ids,
                    calibration=fitted,
                    validation=(matrices[held_out],),
                    noise_scales=bank.noise_scales,
                    column_weights=bank.column_weights,
                    metadata=bank.metadata,
                )
            )
        self.folds = tuple(folds)

    def net_bits(
        self,
        groups: Mapping[str, Sequence[str]],
        closure: Mapping[str, set[str]],
        edge_count: int,
    ) -> float:
        scores = [
            score_circuit(
                fold,
                groups,
                closure,
                edge_count=edge_count,
            )
            for fold in self.folds
        ]
        # Cross-validation describes one circuit and multiple held-out
        # conditions, so predictive evidence accumulates but the circuit is
        # encoded only once.
        return math.fsum(score.predictive_bits for score in scores) - scores[
            0
        ].description_bits


def _mean_calibration(bank: ResponseBank) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            fmean(matrix[source][destination] for matrix in bank.calibration)
            for destination in range(len(bank.feature_ids) + 1)
        )
        for source in range(len(bank.feature_ids))
    )


def _response_signatures(bank: ResponseBank) -> tuple[tuple[float, ...], ...]:
    """Whitened outgoing and incoming causal roles for every feature."""

    mean = _mean_calibration(bank)
    count = len(bank.feature_ids)
    signatures: list[tuple[float, ...]] = []
    for feature in range(count):
        outgoing = [
            mean[feature][destination]
            / bank.noise_scales[feature][destination]
            * math.sqrt(bank.column_weights[destination])
            for destination in range(count + 1)
        ]
        incoming = [
            mean[source][feature]
            / bank.noise_scales[source][feature]
            * math.sqrt(bank.column_weights[feature])
            for source in range(count)
        ]
        signatures.append(tuple(outgoing + incoming))
    return tuple(signatures)


def _centroid(
    cluster: Sequence[int], signatures: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    return tuple(
        fmean(signatures[index][column] for index in cluster)
        for column in range(len(signatures[0]))
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _cluster_distance(
    left: Sequence[int],
    right: Sequence[int],
    signatures: Sequence[Sequence[float]],
) -> float:
    return _distance(_centroid(left, signatures), _centroid(right, signatures))


def _closest_pair(
    clusters: Sequence[Sequence[int]], signatures: Sequence[Sequence[float]]
) -> tuple[int, int]:
    return min(
        (
            (_cluster_distance(clusters[i], clusters[j], signatures), i, j)
            for i in range(len(clusters))
            for j in range(i + 1, len(clusters))
        ),
        key=lambda item: item,
    )[1:]


def _cluster_path(
    selected: Sequence[int],
    signatures: Sequence[Sequence[float]],
    min_group_size: int,
    max_groups: int,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Create valid groups first, then agglomerate to every smaller group count."""

    remaining = list(sorted(selected))
    clusters: list[list[int]] = []
    while len(remaining) >= min_group_size:
        singleton_clusters = [[index] for index in remaining]
        left, right = _closest_pair(singleton_clusters, signatures)
        cluster = [remaining[right], remaining[left]]
        del remaining[right]
        del remaining[left]
        while len(cluster) < min_group_size:
            center = _centroid(cluster, signatures)
            nearest = min(
                remaining,
                key=lambda index: (_distance(center, signatures[index]), index),
            )
            remaining.remove(nearest)
            cluster.append(nearest)
        clusters.append(sorted(cluster))

    for index in remaining:
        destination = min(
            range(len(clusters)),
            key=lambda candidate: (
                _distance(
                    signatures[index], _centroid(clusters[candidate], signatures)
                ),
                min(clusters[candidate]),
            ),
        )
        clusters[destination].append(index)
        clusters[destination].sort()

    path: dict[int, tuple[tuple[int, ...], ...]] = {}
    while clusters:
        clusters.sort(key=lambda cluster: min(cluster))
        if len(clusters) <= max_groups:
            path[len(clusters)] = tuple(tuple(cluster) for cluster in clusters)
        if len(clusters) == 1:
            break
        left, right = _closest_pair(clusters, signatures)
        merged = sorted(clusters[left] + clusters[right])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (left, right)
        ]
        clusters.append(merged)
    return path


def _named_groups(
    clusters: Sequence[Sequence[int]], bank: ResponseBank
) -> dict[str, tuple[str, ...]]:
    ordered = sorted(clusters, key=lambda cluster: min(cluster))
    return {
        f"supernode_{number:02d}": tuple(bank.feature_ids[index] for index in cluster)
        for number, cluster in enumerate(ordered, 1)
    }


def _closure(
    names: Sequence[str], edges: Sequence[tuple[str, str]]
) -> dict[str, set[str]] | None:
    adjacency = {name: set() for name in names}
    for source, target in edges:
        adjacency[source].add(target)
    closure: dict[str, set[str]] = {}
    for start in names:
        reached: set[str] = set()
        stack = list(adjacency[start])
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adjacency.get(node, ()))
        if start in reached:
            return None
        closure[start] = reached
    return closure


def _block_strength(
    bank: ResponseBank,
    sources: Sequence[int],
    destinations: Sequence[int] | None,
) -> float:
    output = len(bank.feature_ids)
    target_indices = (output,) if destinations is None else destinations
    values = [
        matrix[source][destination]
        / bank.noise_scales[source][destination]
        * math.sqrt(bank.column_weights[destination])
        for matrix in bank.calibration
        for source in sources
        for destination in target_indices
    ]
    return abs(fmean(values))


def _induce_topology(
    bank: ResponseBank,
    groups: Mapping[str, Sequence[str]],
    objective: _CalibrationObjective,
) -> tuple[tuple[tuple[str, str], ...], dict[str, set[str]], float]:
    names = list(groups)
    feature_index = {
        feature_id: index for index, feature_id in enumerate(bank.feature_ids)
    }
    indices = {
        name: tuple(feature_index[feature_id] for feature_id in members)
        for name, members in groups.items()
    }
    candidates: list[tuple[float, str, str]] = []
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            forward = _block_strength(bank, indices[left], indices[right])
            backward = _block_strength(bank, indices[right], indices[left])
            if forward >= backward:
                candidates.append((forward, left, right))
            else:
                candidates.append((backward, right, left))
    candidates.extend(
        (_block_strength(bank, indices[name], None), name, OUTPUT) for name in names
    )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    edges: list[tuple[str, str]] = []
    closure = {name: set() for name in names}
    best = objective.net_bits(groups, closure, 0)
    for _, source, target in candidates:
        trial_edges = edges + [(source, target)]
        trial_closure = _closure(names, trial_edges)
        if trial_closure is None:
            continue
        trial = objective.net_bits(groups, trial_closure, len(trial_edges))
        if trial > best + 1e-9:
            edges = trial_edges
            closure = trial_closure
            best = trial
    return tuple(edges), closure, best


def _proposal(candidate: _Candidate) -> dict[str, list[Any]]:
    return {
        "supernodes": [
            {"name": name, "nodes": list(members)}
            for name, members in candidate.groups.items()
        ],
        "edges": [list(edge) for edge in candidate.edges],
    }


def discover_supernodes(
    response_bank: Mapping[str, Any] | ResponseBank,
    *,
    max_groups: int = MAX_GROUPS,
    min_group_size: int = 2,
    select_features: bool = True,
) -> dict[str, Any]:
    """Discover and score a circuit without any natural-language prompt.

    Group membership, topology, and group count are chosen from calibration
    responses. Feature count is selected too unless ``select_features`` is false.
    Validation responses cannot affect the proposal; they are scored exactly once
    after selection.
    """

    bank = (
        response_bank
        if isinstance(response_bank, ResponseBank)
        else ResponseBank.from_dict(response_bank)
    )
    if not 1 <= max_groups <= MAX_GROUPS:
        raise ValueError(f"max_groups must be between 1 and {MAX_GROUPS}")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2")
    if len(bank.feature_ids) < min_group_size:
        raise ValueError("response bank is smaller than min_group_size")

    objective = _CalibrationObjective(bank)
    signatures = _response_signatures(bank)
    energy_order = sorted(
        range(len(bank.feature_ids)),
        key=lambda index: (
            -math.fsum(value * value for value in signatures[index]),
            bank.feature_ids[index],
        ),
    )

    best: _Candidate | None = None
    evaluated = 0
    feature_counts: Sequence[int] = (
        tuple(range(min_group_size, len(bank.feature_ids) + 1))
        if select_features
        else (len(bank.feature_ids),)
    )
    for feature_count in feature_counts:
        selected = energy_order[:feature_count]
        paths = _cluster_path(
            selected,
            signatures,
            min_group_size=min_group_size,
            max_groups=max_groups,
        )
        for group_count in sorted(paths):
            groups = _named_groups(paths[group_count], bank)
            edges, closure, selection_net_bits = _induce_topology(
                bank, groups, objective
            )
            candidate = _Candidate(groups, edges, closure, selection_net_bits)
            evaluated += 1
            if best is None or (
                candidate.selection_net_bits,
                -sum(map(len, candidate.groups.values())),
                -len(candidate.groups),
            ) > (
                best.selection_net_bits,
                -sum(map(len, best.groups.values())),
                -len(best.groups),
            ):
                best = candidate

    if best is None:  # guarded by the size checks, retained as an invariant
        raise RuntimeError("no valid supernode candidate was generated")

    validation: MDLScore = score_circuit(
        bank,
        best.groups,
        best.closure,
        edge_count=len(best.edges),
    )
    return {
        "proposal": _proposal(best),
        "selection": {
            "source": "calibration-only",
            "mode": objective.mode,
            "net_bits": best.selection_net_bits,
            "feature_count": sum(map(len, best.groups.values())),
            "group_count": len(best.groups),
            "edge_count": len(best.edges),
            "candidates_evaluated": evaluated,
            "feature_selection": select_features,
        },
        "validation": validation.to_dict(),
        "bank_id": bank.metadata.get("bank_id"),
        "prompt_or_model_calls": 0,
    }
