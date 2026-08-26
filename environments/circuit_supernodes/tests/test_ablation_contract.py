from circuit_supernodes.runtime.ablation_contract import (
    ablation_result,
    evaluate_cohorts,
    invalid_coordinates,
)


def test_ablation_result_uses_one_positive_and_control_wire_contract() -> None:
    def decode(token_id: int) -> str:
        return f"token-{token_id}"

    positive = ablation_result(
        expected_token="answer",
        expected_id=2,
        baseline_top=2,
        intervened_top=3,
        decode=decode,
        positive=True,
        intervened_expected_probability=0.1,
    )
    control = ablation_result(
        expected_token="answer",
        expected_id=2,
        baseline_top=2,
        intervened_top=2,
        decode=decode,
        positive=False,
    )
    assert positive["changed_top_token"] is True
    assert positive["intervened_expected_probability"] == 0.1
    assert control["preserved_top_token"] is True
    assert positive["baseline_top_token"] == "token-2"


def test_evaluate_cohorts_preserves_roles_and_order() -> None:
    result = evaluate_cohorts(
        [{"id": 1}],
        [{"id": 2}],
        lambda example, *, positive: {"id": example["id"], "positive": positive},
    )
    assert result == {
        "positives": [{"id": 1, "positive": True}],
        "controls": [{"id": 2, "positive": False}],
    }


def test_invalid_coordinates_uses_layer_feature_position_order() -> None:
    assert invalid_coordinates(
        [(0, 1, 2), (1, 0, 0), (0, 2, 0), (0, 0, 3)],
        layers=1,
        features=2,
        positions=3,
    ) == [(1, 0, 0), (0, 2, 0), (0, 0, 3)]
