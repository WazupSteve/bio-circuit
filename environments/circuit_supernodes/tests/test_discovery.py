from copy import deepcopy

from circuit_supernodes.discovery import discover_supernodes
from circuit_supernodes.mdl import ResponseBank


def planted_bank(validation_scale: float = 1.0) -> ResponseBank:
    features = ["a1", "a2", "b1", "b2", "c1", "c2"]
    matrix = [[0.0] * 7 for _ in features]
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
    validation = [
        [value * validation_scale for value in row] for row in deepcopy(matrix)
    ]
    return ResponseBank.from_dict(
        {
            "feature_ids": features,
            "calibration": [matrix, matrix],
            "validation": [validation],
            "noise_scales": [1.0] * 7,
            "column_weights": [1.0] * 7,
            "metadata": {"bank_id": "planted"},
        }
    )


def partition(result: dict) -> set[frozenset[str]]:
    return {
        frozenset(group["nodes"]) for group in result["proposal"]["supernodes"]
    }


def test_prompt_free_discovery_recovers_planted_causal_roles() -> None:
    result = discover_supernodes(planted_bank())
    assert partition(result) == {
        frozenset(("a1", "a2")),
        frozenset(("b1", "b2")),
        frozenset(("c1", "c2")),
    }
    assert result["selection"]["group_count"] == 3
    assert result["selection"]["feature_count"] == 6
    assert result["selection"]["feature_selection"] is True
    assert result["validation"]["circuit_error"] == 0.0
    assert result["prompt_or_model_calls"] == 0


def test_validation_values_cannot_change_discovered_proposal() -> None:
    baseline = discover_supernodes(planted_bank(validation_scale=1.0))
    changed = discover_supernodes(planted_bank(validation_scale=-3.0))
    assert baseline["proposal"] == changed["proposal"]
    assert baseline["selection"] == changed["selection"]
    assert baseline["validation"] != changed["validation"]


def test_prompt_free_discovery_is_deterministic() -> None:
    bank = planted_bank()
    assert discover_supernodes(bank) == discover_supernodes(bank)
