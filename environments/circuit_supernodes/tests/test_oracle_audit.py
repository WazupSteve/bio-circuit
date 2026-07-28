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
    coherence_matched,
    community,
    experiment_cost,
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


def test_coherence_matched_groups_members_with_matching_effects():
    # Both halves of any split must look alike, which is what defeats the
    # geometric mean. Members are contiguous in the measured effect ranking.
    signatures = {f: i / 21 for i, f in enumerate(sorted(FEATURE_IDS))}
    sns = coherence_matched(GRAPH, signatures, 3, MIN_GROUP_SIZE)
    assert len(sns) >= 2
    for ms in sns.values():
        spread = max(signatures[m] for m in ms) - min(signatures[m] for m in ms)
        assert spread < 0.5  # tight in effect space, unlike a blind partition
    parsed, reason = validate_members(as_proposal(sns), GRAPH)
    assert parsed is not None, reason


def test_coherence_matched_respects_the_minimum_size_gate():
    signatures = {f: i / 21 for i, f in enumerate(sorted(FEATURE_IDS))}
    for size in (2, 4, MIN_GROUP_SIZE):
        sns = coherence_matched(GRAPH, signatures, 8, size)
        assert all(len(ms) >= size for ms in sns.values())


def test_signature_attack_is_opt_in():
    assert "coherence-matched" not in attacks(GRAPH, 3, MIN_GROUP_SIZE, 2)
    assert "coherence-matched" in attacks(
        GRAPH, 3, MIN_GROUP_SIZE, 2, dict.fromkeys(FEATURE_IDS, 0.5)
    )


def test_experiment_cost_separates_reachable_attacks_from_oracle_fantasies():
    from circuit_supernodes.experiments import DEFAULT_BUDGET, MAX_ABLATED_FEATURES

    # Five groups of twelve is one ablation each — inside the budget, so this
    # attack is something a policy can actually run, not an upper bound.
    affordable = {f"g{i}": [f"f{i}_{j}" for j in range(12)] for i in range(5)}
    assert experiment_cost(affordable) == 5 <= DEFAULT_BUDGET

    too_wide = {
        "a": [f"f{j}" for j in range(MAX_ABLATED_FEATURES + 1)],
        "b": ["x", "y"],
    }
    assert experiment_cost(too_wide) is None  # tool cannot ablate that many

    assert experiment_cost({"only": ["a", "b"]}) is None  # needs two groups

    # Sixty singleton groups (the per-feature signatures) blow every limit.
    assert experiment_cost({f"g{i}": [f"f{i}"] for i in range(60)}) is None


def test_community_falls_back_when_propagation_collapses():
    # A densely connected graph drives label propagation to a single cluster;
    # without a fallback the attack silently reports "gated" and looks harmless.
    sns = community(GRAPH, 3, MIN_GROUP_SIZE)
    assert len(sns) >= 2
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


def test_reward_does_not_yet_prefer_the_true_partition():
    """Characterization test for a known, unfixed defect.

    With a ground-truth causal structure and the best edge set an exhaustive DAG
    search can find, a blind shuffle scores at or above the true feature
    grouping. Separation rewards groups whose activations respond to many
    ablations in a graded way, and a mixture does that better than a
    functionally pure group — so the metric is biased toward exactly the
    partitions a supernode reward should reject.

    If a reward change makes this fail, that is the fix landing: delete the
    xfail-style assertion below and assert the true partition wins instead.
    """
    from partition_discrimination import _shuffle, best_edges, build

    roles, features, drives, graph = build(3, 4)

    async def run():
        true = await best_edges(graph, roles, drives, dict(roles), True)
        blind = [
            await best_edges(graph, roles, drives, _shuffle(features, 3, s), True)
            for s in range(3)
        ]
        return true, blind

    true, blind = asyncio.run(run())
    assert max(blind) >= true, (
        f"the reward now prefers the true partition (true={true:.3f}, "
        f"blind={blind}) — update this test, the defect is fixed"
    )
