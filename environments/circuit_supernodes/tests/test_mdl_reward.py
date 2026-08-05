from __future__ import annotations

import asyncio
import random

import pytest
from circuit_supernodes.graphs import Graph, GraphNode
from circuit_supernodes.mdl import (
    BIT_SCALE,
    InvalidResponseBank,
    ResponseBank,
    exponential_reward,
    literal_description_bits,
    score_circuit,
)
from circuit_supernodes.rubric import (
    InterventionScoring,
    validate_members,
    validate_topology,
)

FEATURES = ("a1", "a2", "b1", "b2", "c1", "c2")
TRUE_GROUPS = {
    "A": ("a1", "a2"),
    "B": ("b1", "b2"),
    "C": ("c1", "c2"),
}
TRUE_CLOSURE = {
    "A": {"B", "C", "output"},
    "B": {"C", "output"},
    "C": {"output"},
}


def planted_bank(validation_matrices: int = 8) -> ResponseBank:
    """Three two-member causal roles connected as A -> B -> C -> output."""

    n = len(FEATURES)
    matrix = [[0.0] * (n + 1) for _ in range(n)]
    for source in (0, 1):
        for destination in (2, 3):
            matrix[source][destination] = 2.0
        for destination in (4, 5):
            matrix[source][destination] = 1.0
        matrix[source][-1] = 1.0
    for source in (2, 3):
        for destination in (4, 5):
            matrix[source][destination] = 3.0
        matrix[source][-1] = 2.0
    for source in (4, 5):
        matrix[source][-1] = 4.0
    return ResponseBank.from_dict(
        {
            "feature_ids": FEATURES,
            "calibration": [matrix, matrix],
            "validation": [matrix] * validation_matrices,
            "noise_scales": [1.0] * (n + 1),
            "column_weights": [1.0] * (n + 1),
            "metadata": {"bank_id": "planted-chain"},
        }
    )


def planted_score(
    groups=TRUE_GROUPS, closure=TRUE_CLOSURE, edge_count: int = 3, *, repeats: int = 8
):
    return score_circuit(planted_bank(repeats), groups, closure, edge_count=edge_count)


def test_literal_code_charges_every_extra_member_and_edge() -> None:
    base = literal_description_bits(
        universe_size=60, selected_count=4, group_count=2, edge_count=1
    )
    with_member = literal_description_bits(
        universe_size=60, selected_count=5, group_count=2, edge_count=1
    )
    with_edge = literal_description_bits(
        universe_size=60, selected_count=4, group_count=2, edge_count=2
    )
    assert with_member > base
    assert with_edge > base
    assert (
        literal_description_bits(
            universe_size=6, selected_count=6, group_count=3, edge_count=3
        )
        == 53
    )


def test_block_fit_holds_out_the_source_member() -> None:
    bank = ResponseBank.from_dict(
        {
            "feature_ids": ["a", "b"],
            "calibration": [[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]],
            "validation": [[[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]],
            "noise_scales": [1.0, 1.0, 1.0],
            "column_weights": [0.0, 0.0, 1.0],
        }
    )
    score = score_circuit(
        bank,
        {"role": ("a", "b")},
        {"role": {"output"}},
        edge_count=1,
    )
    assert score.circuit_error == pytest.approx(0.0)


def test_exponential_reward_preserves_net_code_saving_threshold() -> None:
    assert exponential_reward(-1.0) == pytest.approx(0.25)
    assert exponential_reward(0.0) == pytest.approx(0.5)
    assert exponential_reward(1.0) == pytest.approx(1.0)
    assert exponential_reward(-2000.0) > 0.0

    for repeats in (1, 2, 8):
        result = planted_score(repeats=repeats)
        below_null = (
            result.circuit_error + BIT_SCALE * result.description_bits
            < result.null_error
        )
        assert (result.reward > 0.5) is below_null
        assert result.reward == pytest.approx(2.0 ** (result.net_reward - 1.0))
        assert result.net_bits == pytest.approx(
            (result.null_error - result.circuit_error) / BIT_SCALE
            - result.description_bits
        )
        assert 0.0 < result.reward <= 1.0


def test_planted_circuit_beats_named_attack_suite() -> None:
    truth = planted_score()
    attacks = {
        "all-feature catch-all": planted_score(
            {"all": FEATURES}, {"all": {"output"}}, 1
        ),
        "tiny strongest-looking subset": planted_score(
            {"C": ("c1", "c2")}, {"C": {"output"}}, 1
        ),
        "random mixed supernodes": planted_score(
            {
                "X": ("a1", "b1"),
                "Y": ("a2", "c1"),
                "Z": ("b2", "c2"),
            },
            {
                "X": {"Y", "Z", "output"},
                "Y": {"Z", "output"},
                "Z": {"output"},
            },
            3,
        ),
        "correct groups with wrong edges": planted_score(
            TRUE_GROUPS,
            {"A": {"output"}, "B": {"output"}, "C": {"output"}},
            3,
        ),
    }
    assert truth.circuit_error == pytest.approx(0.0)
    for attack in attacks.values():
        assert truth.reward > attack.reward
        assert truth.net_bits > attack.net_bits


def test_sixty_four_size_matched_random_partitions_lose() -> None:
    truth = planted_score()
    true_partition = {frozenset(members) for members in TRUE_GROUPS.values()}
    checked = 0
    for seed in range(64):
        shuffled = list(FEATURES)
        random.Random(seed).shuffle(shuffled)
        groups = {
            name: tuple(shuffled[2 * i : 2 * i + 2])
            for i, name in enumerate(("X", "Y", "Z"))
        }
        if {frozenset(members) for members in groups.values()} == true_partition:
            continue
        closure = {
            "X": {"Y", "Z", "output"},
            "Y": {"Z", "output"},
            "Z": {"output"},
        }
        attack = planted_score(groups, closure, 3)
        assert truth.reward > attack.reward
        checked += 1
    assert checked >= 50


def test_every_proper_collection_of_correct_supernodes_loses() -> None:
    truth = planted_score()
    proper_circuits = [
        (("A",), {"A": {"output"}}, 1),
        (("B",), {"B": {"output"}}, 1),
        (("C",), {"C": {"output"}}, 1),
        (("A", "B"), {"A": {"B", "output"}, "B": {"output"}}, 2),
        (("A", "C"), {"A": {"C", "output"}, "C": {"output"}}, 2),
        (("B", "C"), {"B": {"C", "output"}, "C": {"output"}}, 2),
    ]
    for names, closure, edge_count in proper_circuits:
        groups = {name: TRUE_GROUPS[name] for name in names}
        partial = planted_score(groups, closure, edge_count)
        assert truth.reward > partial.reward
        assert truth.net_bits > partial.net_bits


def test_redundant_transitive_edge_only_adds_code_cost() -> None:
    minimal = planted_score(edge_count=3)
    redundant = planted_score(edge_count=4)
    assert redundant.circuit_error == pytest.approx(minimal.circuit_error)
    assert redundant.description_bits > minimal.description_bits
    assert redundant.net_bits < minimal.net_bits
    assert redundant.reward < minimal.reward


def test_response_bank_validation_rejects_bad_scales() -> None:
    matrix = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    with pytest.raises(InvalidResponseBank, match="positive"):
        ResponseBank.from_dict(
            {
                "feature_ids": ["a", "b"],
                "calibration": [matrix],
                "validation": [matrix],
                "noise_scales": [1.0, 0.0, 1.0],
                "column_weights": [1.0, 1.0, 1.0],
            }
        )


def test_strict_validation_rejects_duplicate_edges_but_not_catch_all() -> None:
    graph = Graph(tokens=[], top_outputs=[("x", 0.5)])
    graph.nodes = {
        feature_id: GraphNode(feature_id, "feature", 0, 0, "", 0.0, 1.0)
        for feature_id in FEATURES
    }
    proposal = {
        "supernodes": [{"name": "all", "nodes": list(FEATURES)}],
        "edges": [["all", "output"]],
    }
    members, reason = validate_members(proposal, graph)
    assert reason == ""
    assert members == {"all": list(FEATURES)}
    closure, reason = validate_topology(["all"], [["all", "output"], ["ALL", "OUTPUT"]])
    assert closure is None
    assert "duplicate edge" in reason


def test_environment_scorer_uses_only_mdl_components() -> None:
    graph = Graph(tokens=[], top_outputs=[("target", 0.5)])
    graph.nodes = {
        feature_id: GraphNode(feature_id, "feature", 0, 0, "", 0.0, 1.0)
        for feature_id in FEATURES
    }
    proposal = {
        "supernodes": [
            {"name": name, "nodes": list(members)}
            for name, members in TRUE_GROUPS.items()
        ],
        "edges": [["A", "B"], ["B", "C"], ["C", "output"]],
    }
    result = asyncio.run(
        InterventionScoring().score(
            "prompt", graph, proposal, response_bank=planted_bank()
        )
    )
    assert result["reward_version"] == "v4.0-exponential-mdl"
    assert result["bank_id"] == "planted-chain"
    assert result["circuit_error"] == pytest.approx(0.0)
    assert "separation" not in result
    assert "coverage" not in result
    assert "completeness" not in result


def test_every_pair_partition_is_finite_and_bounded() -> None:
    """Exhaust the 15 unlabeled pairings of six features as a small-world check."""

    def pairings(values: tuple[str, ...]):
        if not values:
            yield ()
            return
        first = values[0]
        for i in range(1, len(values)):
            pair = (first, values[i])
            rest = values[1:i] + values[i + 1 :]
            for suffix in pairings(rest):
                yield (pair,) + suffix

    scores = []
    for pairing in pairings(FEATURES):
        groups = {
            name: members
            for name, members in zip(("X", "Y", "Z"), pairing, strict=True)
        }
        closure = {
            "X": {"Y", "Z", "output"},
            "Y": {"Z", "output"},
            "Z": {"output"},
        }
        score = planted_score(groups, closure, 3)
        assert 0.0 < score.reward <= 1.0
        scores.append(score)
    assert len(scores) == 15
