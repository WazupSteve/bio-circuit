"""Backend-independent request/result contract for Modal ablation services."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def invalid_coordinates(
    coordinates: Sequence[tuple[int, int, int]],
    *,
    layers: int,
    positions: int,
    features: int,
) -> list[tuple[int, int, int]]:
    """Return coordinates outside a layer/feature/position activation tensor."""

    return [
        coordinate
        for coordinate in coordinates
        if not 0 <= coordinate[0] < layers
        or not 0 <= coordinate[1] < features
        or not 0 <= coordinate[2] < positions
    ]


def ablation_result(
    *,
    expected_token: str,
    expected_id: int,
    baseline_top: int,
    intervened_top: int,
    decode: Callable[[int], str],
    positive: bool,
    intervened_expected_probability: float | None = None,
) -> dict[str, Any]:
    """Build the one canonical wire result for positive or control replay."""

    result: dict[str, Any] = {
        "expected_token": expected_token,
        "baseline_expected_is_top": baseline_top == expected_id,
        "baseline_top_token": decode(baseline_top),
        "intervened_expected_is_top": intervened_top == expected_id,
        "intervened_top_token": decode(intervened_top),
    }
    if intervened_expected_probability is not None:
        result["intervened_expected_probability"] = intervened_expected_probability
    gate = baseline_top == expected_id
    if positive:
        result["changed_top_token"] = gate and intervened_top != expected_id
    else:
        result["preserved_top_token"] = gate and intervened_top == expected_id
    return result


def evaluate_cohorts(
    positives: Sequence[dict],
    controls: Sequence[dict],
    evaluate: Callable[..., dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Apply one backend evaluator under the canonical cohort envelope."""

    return {
        "positives": [evaluate(example, positive=True) for example in positives],
        "controls": [evaluate(example, positive=False) for example in controls],
    }
