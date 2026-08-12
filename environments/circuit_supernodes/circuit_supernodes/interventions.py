"""Shared validation for feature interventions.

Coordinates use the public graph order ``(layer, feature, position)``.  The
Modal backend converts to circuit-tracer's ``(layer, position, feature, value)``
order only at the final call boundary.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

Coordinate = tuple[int, int, int]
Intervention = tuple[int, int, int, float]

MAX_INTERVENTIONS = 60
MAX_ABS_WEIGHT = 8.0
MIN_CORRUPTION_NODES = 4
MAX_CORRUPTION_NODES = 60
MIN_CORRUPTION_FACTOR = 2.0
MAX_CORRUPTION_FACTOR = 8.0
CORRUPTION_VERSION = "v2-shared-stable-multiplicative-cohort"

_FEATURE_ID = re.compile(r"^(\d+)_(\d+)_(\d+)$")


def coordinate_from_feature_id(node_id: str) -> Coordinate:
    """Parse an exact displayed feature id without accepting extra segments."""

    match = _FEATURE_ID.fullmatch(node_id)
    if match is None:
        raise ValueError(f"invalid FEATURE id {node_id!r}")
    return tuple(map(int, match.groups()))


def _coordinate_value(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    try:
        exact = float(value) == number  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        exact = False
    if not exact or number < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return number


def validate_interventions(
    raw: Sequence[Sequence[object]],
    *,
    label: str,
    require_nonempty: bool = False,
    min_count: int | None = None,
    max_count: int | None = None,
    min_factor: float | None = None,
    max_abs_factor: float | None = None,
    require_positive: bool = False,
) -> list[Intervention]:
    """Normalize a compact intervention payload and reject ambiguous values."""

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{label} must be a list")
    if require_nonempty and not raw:
        raise ValueError(f"{label} must contain at least one item")
    if min_count is not None and len(raw) < min_count:
        raise ValueError(f"{label} must contain at least {min_count} items")
    if max_count is not None and len(raw) > max_count:
        raise ValueError(f"{label} must contain at most {max_count} items")

    normalized: list[Intervention] = []
    seen: set[Coordinate] = set()
    for index, item in enumerate(raw):
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) != 4
        ):
            raise ValueError(f"{label} item {index} must contain four values")
        coordinate = tuple(
            _coordinate_value(value, f"{label} item {index} coordinate")
            for value in item[:3]
        )
        if coordinate in seen:
            raise ValueError(f"{label} repeats coordinate {coordinate}")
        factor_value = item[3]
        if isinstance(factor_value, bool):
            raise ValueError(f"{label} factor at {coordinate} must be numeric")
        try:
            factor = float(factor_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} factor at {coordinate} must be numeric") from exc
        if not math.isfinite(factor):
            raise ValueError(f"{label} factor at {coordinate} must be finite")
        if require_positive and factor <= 0.0:
            raise ValueError(f"{label} factor at {coordinate} must be positive")
        if min_factor is not None and factor < min_factor:
            raise ValueError(
                f"{label} factor at {coordinate} must be at least {min_factor:g}"
            )
        if max_abs_factor is not None and abs(factor) > max_abs_factor:
            raise ValueError(
                f"{label} factor at {coordinate} must lie in "
                f"[-{max_abs_factor:g}, {max_abs_factor:g}]"
            )
        seen.add(coordinate)
        normalized.append((*coordinate, factor))
    return normalized


def compose_multipliers(
    corruption: Sequence[Intervention], restorations: Sequence[Intervention]
) -> dict[Coordinate, float]:
    """Compose hidden and submitted factors at each affected coordinate."""

    factors = {item[:3]: item[3] for item in corruption}
    for layer, feature, position, weight in restorations:
        coordinate = (layer, feature, position)
        factors[coordinate] = factors.get(coordinate, 1.0) * weight
    return factors
