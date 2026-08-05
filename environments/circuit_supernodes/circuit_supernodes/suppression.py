"""Paired causal-profile analysis for feature suppression in a provided model."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean, stdev
from typing import Any


class InvalidFeatureProfile(ValueError):
    """A model profile cannot be aligned safely with its reference profile."""


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidFeatureProfile(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise InvalidFeatureProfile(f"{label} must be finite")
    return number


def _profile_index(
    profile: Mapping[str, Any], label: str
) -> tuple[str, str, dict[str, str], dict[tuple[str, str], dict[str, Any]]]:
    if not isinstance(profile, Mapping):
        raise InvalidFeatureProfile(f"{label} profile must be an object")
    model = str(profile.get("model", "")).strip()
    transcoders = str(profile.get("transcoders", "")).strip()
    if not model or not transcoders:
        raise InvalidFeatureProfile(f"{label} profile needs model and transcoders")
    records = profile.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise InvalidFeatureProfile(f"{label} records must be a list")

    fingerprints: dict[str, str] = {}
    measurements: dict[tuple[str, str], dict[str, Any]] = {}
    for record_number, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise InvalidFeatureProfile(f"{label} record {record_number} is invalid")
        prompt_id = str(record.get("prompt_id", "")).strip()
        fingerprint = str(record.get("token_fingerprint", "")).strip()
        if not prompt_id or not fingerprint:
            raise InvalidFeatureProfile(
                f"{label} record {record_number} needs prompt_id and token_fingerprint"
            )
        if prompt_id in fingerprints and fingerprints[prompt_id] != fingerprint:
            raise InvalidFeatureProfile(f"{label} repeats {prompt_id!r} inconsistently")
        fingerprints[prompt_id] = fingerprint
        raw_measurements = record.get("measurements")
        if not isinstance(raw_measurements, Sequence) or isinstance(
            raw_measurements, (str, bytes)
        ):
            raise InvalidFeatureProfile(
                f"{label} record {prompt_id!r} measurements must be a list"
            )
        for measurement in raw_measurements:
            if not isinstance(measurement, Mapping):
                raise InvalidFeatureProfile(
                    f"{label} record {prompt_id!r} has an invalid measurement"
                )
            feature_id = str(measurement.get("feature_id", "")).strip()
            feature_key = str(measurement.get("feature_key", "")).strip()
            if not feature_id or not feature_key:
                raise InvalidFeatureProfile(
                    f"{label} measurement needs feature_id and feature_key"
                )
            key = (prompt_id, feature_id)
            if key in measurements:
                raise InvalidFeatureProfile(f"{label} repeats measurement {key!r}")
            measurements[key] = {
                "feature_key": feature_key,
                "support": _number(
                    measurement.get("mean_clamp_support_log_odds"),
                    f"{label} {key!r} support",
                ),
                "activation_excess": _number(
                    measurement.get("activation_excess"),
                    f"{label} {key!r} activation_excess",
                ),
            }
    return model, transcoders, fingerprints, measurements


def compare_feature_profiles(
    reference: Mapping[str, Any],
    provided: Mapping[str, Any],
    *,
    min_occurrences: int = 2,
    min_reference_support: float = 0.05,
    min_absolute_suppression: float = 0.05,
    min_suppression_fraction: float = 0.25,
    min_consistency: float = 0.6,
) -> dict[str, Any]:
    """Rank reference-supporting features suppressed in the provided model.

    Suppression means that clamping a feature from its prompt-position activation
    to its across-position mean removes less target log-odds in the provided model
    than in the reference. Only exactly aligned prompt tokens and transcoder
    coordinates are compared.
    """

    if min_occurrences < 1:
        raise ValueError("min_occurrences must be positive")
    if not 0.0 <= min_consistency <= 1.0:
        raise ValueError("min_consistency must lie in [0, 1]")
    if min_reference_support < 0 or min_absolute_suppression < 0:
        raise ValueError("support thresholds must be nonnegative")
    if min_suppression_fraction < 0:
        raise ValueError("min_suppression_fraction must be nonnegative")

    (
        reference_model,
        reference_transcoders,
        reference_fingerprints,
        reference_measurements,
    ) = _profile_index(reference, "reference")
    (
        provided_model,
        provided_transcoders,
        provided_fingerprints,
        provided_measurements,
    ) = _profile_index(provided, "provided")
    if reference_transcoders != provided_transcoders:
        raise InvalidFeatureProfile(
            "profiles use different transcoders; feature coordinates are not aligned"
        )

    common_prompts = set(reference_fingerprints) & set(provided_fingerprints)
    if not common_prompts:
        raise InvalidFeatureProfile("profiles have no aligned prompt ids")
    for prompt_id in sorted(common_prompts):
        if reference_fingerprints[prompt_id] != provided_fingerprints[prompt_id]:
            raise InvalidFeatureProfile(
                f"tokenization differs for prompt {prompt_id!r}; comparison is unsafe"
            )

    aligned_keys = sorted(
        set(reference_measurements) & set(provided_measurements)
    )
    if not aligned_keys:
        raise InvalidFeatureProfile("profiles have no aligned feature measurements")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prompt_id, feature_id in aligned_keys:
        reference_value = reference_measurements[(prompt_id, feature_id)]
        provided_value = provided_measurements[(prompt_id, feature_id)]
        if reference_value["feature_key"] != provided_value["feature_key"]:
            raise InvalidFeatureProfile(
                f"feature identity differs for {(prompt_id, feature_id)!r}"
            )
        grouped[reference_value["feature_key"]].append(
            {
                "prompt_id": prompt_id,
                "feature_id": feature_id,
                "reference_support_log_odds": reference_value["support"],
                "provided_support_log_odds": provided_value["support"],
                "support_delta": reference_value["support"]
                - provided_value["support"],
                "reference_activation_excess": reference_value["activation_excess"],
                "provided_activation_excess": provided_value["activation_excess"],
            }
        )

    suppressed: list[dict[str, Any]] = []
    features_considered = 0
    for feature_key, evidence in sorted(grouped.items()):
        if len(evidence) < min_occurrences:
            continue
        features_considered += 1
        reference_support = fmean(
            item["reference_support_log_odds"] for item in evidence
        )
        provided_support = fmean(
            item["provided_support_log_odds"] for item in evidence
        )
        deltas = [item["support_delta"] for item in evidence]
        suppression = fmean(deltas)
        fraction = suppression / max(abs(reference_support), 1e-12)
        consistency = sum(delta > 0 for delta in deltas) / len(deltas)
        if not (
            reference_support >= min_reference_support
            and suppression >= min_absolute_suppression
            and fraction >= min_suppression_fraction
            and consistency >= min_consistency
        ):
            continue

        reference_activation = fmean(
            abs(item["reference_activation_excess"]) for item in evidence
        )
        provided_activation = fmean(
            abs(item["provided_activation_excess"]) for item in evidence
        )
        activation_reduction = reference_activation - provided_activation
        delta_sd = stdev(deltas) if len(deltas) > 1 else 0.0
        stability = suppression / (
            delta_sd / math.sqrt(len(deltas))
            + max(0.05 * abs(reference_support), 1e-6)
        )
        suppressed.append(
            {
                "feature_key": feature_key,
                "occurrences": len(evidence),
                "reference_support_log_odds": reference_support,
                "provided_support_log_odds": provided_support,
                "support_suppression_log_odds": suppression,
                "support_suppression_fraction": fraction,
                "positive_delta_fraction": consistency,
                "support_delta_sd": delta_sd,
                "stability_score": stability,
                "reference_activation_excess_magnitude": reference_activation,
                "provided_activation_excess_magnitude": provided_activation,
                "activation_reduction": activation_reduction,
                "activation_backed": activation_reduction > 0.0,
                "strongest_evidence": sorted(
                    evidence,
                    key=lambda item: (-item["support_delta"], item["prompt_id"]),
                )[:10],
            }
        )

    suppressed.sort(
        key=lambda item: (
            -item["stability_score"],
            -item["support_suppression_log_odds"],
            item["feature_key"],
        )
    )
    for rank, feature in enumerate(suppressed, 1):
        feature["rank"] = rank

    return {
        "reference_model": reference_model,
        "provided_model": provided_model,
        "transcoders": reference_transcoders,
        "definition": (
            "loss of positive target-log-odds support under an across-position "
            "mean clamp, relative to the aligned reference model"
        ),
        "thresholds": {
            "min_occurrences": min_occurrences,
            "min_reference_support": min_reference_support,
            "min_absolute_suppression": min_absolute_suppression,
            "min_suppression_fraction": min_suppression_fraction,
            "min_consistency": min_consistency,
        },
        "alignment": {
            "common_prompts": len(common_prompts),
            "aligned_measurements": len(aligned_keys),
            "reference_only_measurements": len(
                set(reference_measurements) - set(provided_measurements)
            ),
            "provided_only_measurements": len(
                set(provided_measurements) - set(reference_measurements)
            ),
            "features_considered": features_considered,
        },
        "suppressed_features": suppressed,
    }
