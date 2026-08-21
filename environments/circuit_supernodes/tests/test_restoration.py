import pytest
from circuit_supernodes.grading.restoration import (
    cohort_clean_similarity,
    restoration_similarity,
)
from circuit_supernodes.models import TracerGraphPayload, TracerNode
from circuit_supernodes.runtime.corruption import compose_multipliers
from circuit_supernodes.runtime.graphs import node_by_id
from circuit_supernodes.runtime.responses import render_graph
from scripts.build_restoration_dataset import (
    all_changed,
    common_stable_features,
    coordinate_specs,
)


def graph() -> TracerGraphPayload:
    return TracerGraphPayload(
        metadata={"prompt_tokens": ["test"]},
        nodes=[
            TracerNode(
                node_id="1_2_0",
                feature_type="feature",
                layer=1,
                ctx_idx=0,
                influence=0.4,
            ),
            TracerNode(
                node_id="logit",
                feature_type="logit",
                layer=2,
                ctx_idx=0,
                clerp=" token",
                token_prob=0.5,
                influence=0.4,
            ),
        ],
    )


def test_similarity_is_zero_until_target_is_top() -> None:
    similarity = restoration_similarity(
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_standard_kl": 0.2,
            "restored_target_is_top": False,
        }
    )
    assert similarity == 0.0


def test_metric_uses_clean_distribution_similarity_after_gate() -> None:
    similarity = restoration_similarity(
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_standard_kl": 0.2,
            "restored_target_is_top": True,
        }
    )
    assert similarity == pytest.approx(0.818730753)


def test_cohort_similarity_averages_exact_gated_clean_recovery() -> None:
    results = [
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_target_is_top": True,
            "restored_standard_kl": 0.0,
        },
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_target_is_top": True,
            "restored_standard_kl": 1.0,
        },
        {
            "standard_target_is_top": True,
            "corruption_changed_top_token": True,
            "restored_target_is_top": False,
            "restored_standard_kl": 0.0,
        },
    ]

    expected = (1.0 + 1 / 2.718281828) / 3
    assert cohort_clean_similarity(results) == pytest.approx(expected)


def test_restoration_weights_compose_with_hidden_factors() -> None:
    assert compose_multipliers(
        [(1, 2, 0, 4.0)],
        [(1, 2, 0, 0.25), (3, 4, 0, 2.0)],
    ) == {(1, 2, 0): 1.0, (3, 4, 0): 2.0}


def test_corrupted_activation_survives_storage_and_rendering() -> None:
    subject = graph()
    node_by_id(subject)["1_2_0"].activation = 0.125
    restored = TracerGraphPayload.model_validate(subject.model_dump())
    assert node_by_id(restored)["1_2_0"].activation == 0.125
    assert "+0.125" in render_graph(restored)


def test_builder_intersects_stable_features_across_prompt_positions() -> None:
    first = graph()
    node_by_id(first)["1_2_0"].activation = 0.2
    second = graph()
    second.nodes = [
        TracerNode(
            node_id="1_2_7",
            feature_type="feature",
            layer=1,
            ctx_idx=7,
            influence=0.4,
            activation=0.2,
        )
    ]
    assert common_stable_features([first, second]) == ["1_2"]
    assert all_changed([{"changed_top_token": True}, {"changed_top_token": True}])
    assert not all_changed([{"changed_top_token": True}, {"changed_top_token": False}])


def test_builder_expands_one_stable_corruption_across_prompt_positions() -> None:
    coordinates = coordinate_specs(
        [(1, 2, 4.0), (3, 4, 2.0)],
        [{"1_2": 0, "3_4": 1}, {"1_2": 7, "3_4": 9}],
    )
    assert coordinates == [
        [(1, 2, 0, 4.0), (3, 4, 1, 2.0)],
        [(1, 2, 7, 4.0), (3, 4, 9, 2.0)],
    ]
