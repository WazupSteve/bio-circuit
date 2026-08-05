import pytest
from circuit_supernodes.suppression import (
    InvalidFeatureProfile,
    compare_feature_profiles,
)


def profile(model: str, *, changed: bool, fingerprint: str = "same") -> dict:
    records = []
    for index, (strong, stable, inhibitory) in enumerate(
        ((1.0, 0.4, -0.5), (1.2, 0.5, -0.4), (0.8, 0.45, -0.6))
    ):
        records.append(
            {
                "prompt_id": f"prompt-{index}",
                "token_fingerprint": fingerprint,
                "measurements": [
                    {
                        "feature_id": f"0_10_{index}",
                        "feature_key": "0_10",
                        "mean_clamp_support_log_odds": (
                            strong * 0.2 if changed else strong
                        ),
                        "activation_excess": 0.5 if changed else 2.0,
                    },
                    {
                        "feature_id": f"1_20_{index}",
                        "feature_key": "1_20",
                        "mean_clamp_support_log_odds": stable,
                        "activation_excess": 1.0,
                    },
                    {
                        "feature_id": f"2_30_{index}",
                        "feature_key": "2_30",
                        "mean_clamp_support_log_odds": (
                            inhibitory * 0.1 if changed else inhibitory
                        ),
                        "activation_excess": 0.3 if changed else 1.0,
                    },
                ],
            }
        )
    return {
        "model": model,
        "transcoders": "shared/transcoders",
        "records": records,
    }


def test_discovers_only_lost_positive_causal_support() -> None:
    result = compare_feature_profiles(
        profile("reference", changed=False),
        profile("provided", changed=True),
    )
    assert [item["feature_key"] for item in result["suppressed_features"]] == [
        "0_10"
    ]
    feature = result["suppressed_features"][0]
    assert feature["rank"] == 1
    assert feature["occurrences"] == 3
    assert feature["support_suppression_fraction"] == pytest.approx(0.8)
    assert feature["positive_delta_fraction"] == 1.0
    assert feature["activation_backed"] is True


def test_rejects_tokenization_mismatch() -> None:
    with pytest.raises(InvalidFeatureProfile, match="tokenization differs"):
        compare_feature_profiles(
            profile("reference", changed=False, fingerprint="reference-tokens"),
            profile("provided", changed=True, fingerprint="provided-tokens"),
        )


def test_rejects_unaligned_transcoder_spaces() -> None:
    provided = profile("provided", changed=True)
    provided["transcoders"] = "different/transcoders"
    with pytest.raises(InvalidFeatureProfile, match="different transcoders"):
        compare_feature_profiles(profile("reference", changed=False), provided)


def test_minimum_occurrence_threshold_is_enforced() -> None:
    result = compare_feature_profiles(
        profile("reference", changed=False),
        profile("provided", changed=True),
        min_occurrences=4,
    )
    assert result["suppressed_features"] == []
    assert result["alignment"]["features_considered"] == 0
