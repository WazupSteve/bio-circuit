"""The audit is the Stage 1 kill switch, so its attacks have to be real attacks.

Every partition it builds must clear the same gates a policy's proposal does —
an attack that merely trips a gate proves nothing — and the edge fit has to be
the best claim the measurement supports, not a lucky guess.
"""

import asyncio

import pytest
from circuit_supernodes.rubric import (
    MIN_GROUP_SIZE,
    validate_members,
    validate_topology,
)
from conftest import GRAPH, GROUPS, PROMPT
from oracle_audit import (
    _rebalance,
    attacks,
    best_score,
    blind_balanced,
    community,
    fit_edges,
    random_partition,
    smallest_legal,
)

FEATURE_IDS = {n.id for n in GRAPH.feature_nodes}


def as_proposal(sns: dict, edges=None) -> dict:
    return {
        "supernodes": [{"name": n, "nodes": ms} for n, ms in sns.items()],
        "edges": edges or [[n, "output"] for n in sns],
    }


@pytest.mark.parametrize(
    "build",
    [
        lambda: blind_balanced(GRAPH, 3, MIN_GROUP_SIZE),
        lambda: community(GRAPH, 3, MIN_GROUP_SIZE),
        lambda: random_partition(GRAPH, 3, MIN_GROUP_SIZE, seed=0),
        lambda: smallest_legal(GRAPH, MIN_GROUP_SIZE),
    ],
)
def test_every_attack_clears_the_membership_gate(build):
    sns = build()
    assert len(sns) >= 2
    members = [m for ms in sns.values() for m in ms]
    assert set(members) <= FEATURE_IDS
    assert len(set(members)) == len(members)  # disjoint
    assert all(len(ms) >= MIN_GROUP_SIZE for ms in sns.values())
    parsed, reason = validate_members(as_proposal(sns), GRAPH)
    assert parsed is not None, reason


def test_random_multistart_actually_varies():
    partitions = {
        tuple(
            sorted(
                tuple(sorted(ms))
                for ms in random_partition(GRAPH, 3, MIN_GROUP_SIZE, seed).values()
            )
        )
        for seed in range(8)
    }
    assert len(partitions) > 1


def test_smallest_legal_uses_the_smallest_groups_the_gate_permits():
    for size in (2, 4, MIN_GROUP_SIZE):
        sns = smallest_legal(GRAPH, size)
        assert all(len(ms) == size for ms in sns.values())


def test_rebalance_drops_groups_it_cannot_bring_up_to_size():
    # Nine features cannot make four groups of six; return only legal ones.
    chunks = [[f"f{i}"] for i in range(9)]
    assert all(len(g) >= 6 for g in _rebalance(chunks, 4, 6))


def measurement_for(sns: dict) -> dict:
    """Effects consistent with a chain: each group drives the next and output."""
    names = list(sns)
    order = {n: i for i, n in enumerate(names)}
    return {
        "names": names,
        "target": [0.8 for _ in names],
        "groups": [
            [0.9 if order[b] == order[a] + 1 else 0.0 for b in names] for a in names
        ],
    }


def test_fit_edges_yields_a_dag_that_clears_the_topology_gate():
    sns = blind_balanced(GRAPH, 3, MIN_GROUP_SIZE)
    for threshold in (0.0, 0.5, 0.95, 1.5):
        edges = fit_edges(measurement_for(sns), GRAPH, sns, threshold)
        closure, reason = validate_topology(list(sns), edges)
        assert closure is not None, f"{threshold}: {reason}"
        assert all("output" in c for c in closure.values())


def test_fit_edges_tracks_the_measurement():
    sns = blind_balanced(GRAPH, 3, MIN_GROUP_SIZE)
    measurement = measurement_for(sns)
    names = measurement["names"]
    tight = fit_edges(measurement, GRAPH, sns, 0.5)
    # Only the measured chain steps survive; the rest are forced output links.
    assert [names[0], names[1]] in tight
    assert [names[0], names[2]] not in tight
    loose = fit_edges(measurement, GRAPH, sns, 0.0)
    assert len(loose) > len(tight)


class FakeScorer:
    """Scores a proposal by how well its closure matches a known chain."""

    def __init__(self):
        self.measurements = 0

    async def measure(self, prompt, g, sns):
        self.measurements += 1
        return measurement_for(sns)

    async def score(self, prompt, g, proposal):
        closure, reason = validate_topology(
            [sn["name"] for sn in proposal["supernodes"]], proposal["edges"]
        )
        if closure is None:
            return {"reward": 0.0, "gated": reason}
        # Prefer parsimony: reward falls as the claimed closure grows.
        claimed = sum(len(c) for c in closure.values())
        return {"reward": 1.0 / claimed, "separation": 1.0 / claimed}


def test_best_score_measures_once_and_keeps_the_best_edge_set():
    sns = blind_balanced(GRAPH, 3, MIN_GROUP_SIZE)
    scorer = FakeScorer()
    best = asyncio.run(best_score(scorer, PROMPT, GRAPH, sns))
    assert scorer.measurements == 1  # the sweep is free after one intervention
    assert best["reward"] > 0.0
    assert "threshold" in best


def test_best_score_reports_a_gate_when_an_attack_degenerates():
    result = asyncio.run(best_score(FakeScorer(), PROMPT, GRAPH, {"only": GROUPS["A"]}))
    assert result["reward"] == 0.0 and "too few groups" in result["gated"]


def test_attack_suite_covers_the_four_documented_families():
    suite = attacks(GRAPH, 3, MIN_GROUP_SIZE, multistart=4)
    assert set(suite) == {
        "blind-balanced",
        "community",
        "random-multistart",
        "smallest-legal",
    }
    assert len(suite["random-multistart"]) == 4
