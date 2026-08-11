import pytest
from circuit_supernodes.graphs import (
    Graph,
    GraphNode,
    graph_to_json,
    load_graph_json,
    render_graph,
)
from circuit_supernodes.interventions import (
    MIN_CORRUPTION_NODES,
    compose_multipliers,
)
from circuit_supernodes.restoration import (
    InvalidRestorationResult,
    parse_intervention_graph,
    reward_from_restoration,
    validate_corruption,
    validate_intervention_graph,
)


def graph() -> Graph:
    return Graph(
        tokens=["test"],
        top_outputs=[(" token", 0.5)],
        nodes={
            "1_2_0": GraphNode("1_2_0", "feature", 1, 0, "test", 0.4),
            "logit": GraphNode("logit", "logit", 2, 0, " token", 0.5),
        },
    )


def test_parses_and_validates_weighted_intervention_graph() -> None:
    proposal = parse_intervention_graph(
        '```json\n{"nodes":[{"id":"1_2_0","intervention_weight":0.75}]}\n```'
    )
    assert proposal is not None
    interventions, error = validate_intervention_graph(proposal, graph())
    assert error == ""
    assert interventions == [("1_2_0", 0.75)]


def test_parser_enforces_the_exact_reply_contract() -> None:
    block = '```json\n{"nodes":[{"id":"1_2_0","intervention_weight":1}]}\n```'
    assert parse_intervention_graph(block) is not None
    assert parse_intervention_graph(f"analysis first\n{block}") is None
    assert parse_intervention_graph('{"nodes": []}') is None


@pytest.mark.parametrize(
    ("proposal", "message"),
    [
        ({"nodes": []}, "use 1-60"),
        (
            {"nodes": [{"id": "missing", "intervention_weight": 1.0}]},
            "not an exported FEATURE",
        ),
        (
            {"nodes": [{"id": "1_2_0", "intervention_weight": 9.0}]},
            "must lie in [-8, 8]",
        ),
    ],
)
def test_rejects_invalid_intervention_graph(proposal: dict, message: str) -> None:
    interventions, error = validate_intervention_graph(proposal, graph())
    assert interventions is None
    assert message in error


def test_reward_is_zero_until_target_is_top() -> None:
    reward, mode = reward_from_restoration(
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_standard_kl": 0.2,
            "restored_target_is_top": False,
        }
    )
    assert reward == 0.0
    assert mode == "target_token_gate"


def test_reward_uses_clean_distribution_similarity_after_gate() -> None:
    reward, mode = reward_from_restoration(
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_standard_kl": 0.2,
            "restored_target_is_top": True,
        }
    )
    assert reward == pytest.approx(0.818730753)
    assert mode == "exact_gated_clean_similarity"


@pytest.mark.parametrize(
    "result",
    [
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_target_is_top": True,
        },
        {
            "standard_target_is_top": False,
            "corruption_changed_top_token": True,
            "restored_target_is_top": True,
            "restored_standard_kl": 0.0,
        },
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": False,
            "restored_target_is_top": True,
            "restored_standard_kl": 0.0,
        },
    ],
)
def test_backend_protocol_mismatches_raise_instead_of_earning_reward(
    result: dict,
) -> None:
    with pytest.raises(InvalidRestorationResult):
        reward_from_restoration(result)


def test_hidden_corruption_is_compact_and_nonzero() -> None:
    corruption = [[1, feature, 0, 2.0] for feature in range(MIN_CORRUPTION_NODES)]
    assert validate_corruption(corruption) == [tuple(item) for item in corruption]
    corruption[0][-1] = 0.0
    with pytest.raises(ValueError, match="positive"):
        validate_corruption(corruption)


def test_corruption_rejects_truncated_and_nonintegral_payloads() -> None:
    with pytest.raises(ValueError, match="at least 40"):
        validate_corruption([[1, 2, 0, 2.0]])
    corruption = [[1, feature, 0, 2.0] for feature in range(MIN_CORRUPTION_NODES)]
    corruption[0][0] = 1.5
    with pytest.raises(ValueError, match="nonnegative integer"):
        validate_corruption(corruption)


def test_restoration_weights_compose_with_hidden_factors() -> None:
    assert compose_multipliers(
        [(1, 2, 0, 4.0)],
        [(1, 2, 0, 0.25), (3, 4, 0, 2.0)],
    ) == {(1, 2, 0): 1.0, (3, 4, 0): 2.0}


def test_corrupted_activation_survives_storage_and_rendering() -> None:
    subject = graph()
    subject.nodes["1_2_0"].activation = 0.125
    restored = load_graph_json(graph_to_json(subject))
    assert restored.nodes["1_2_0"].activation == 0.125
    assert "+0.125" in render_graph(restored)
