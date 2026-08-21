"""Shared multiplicative-corruption policy and composition operations."""

from collections.abc import Sequence
from typing import Self

from pydantic import ConfigDict, RootModel, ValidationInfo, model_validator

from ..models import CoordinateTuple, Intervention

MIN_CORRUPTION_NODES = 4
MAX_CORRUPTION_NODES = 60
MIN_CORRUPTION_FACTOR = 0.0
MAX_CORRUPTION_FACTOR = 8.0


class InterventionPayload(RootModel[list[Intervention]]):
    """Strict compact intervention payload shared by corruption services."""

    model_config = ConfigDict(strict=True)

    @model_validator(mode="before")
    @classmethod
    def json_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return [tuple(item) if isinstance(item, list) else item for item in value]
        return value

    @model_validator(mode="after")
    def valid_items(self, info: ValidationInfo) -> Self:
        coordinates = [item[:3] for item in self.root]
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("interventions repeat a coordinate")
        context = info.context or {}
        label = str(context.get("label", "intervention payload"))
        min_count = int(context.get("min_count", 0))
        max_count = context.get("max_count")
        min_factor = float(context.get("min_factor", 0.0))
        max_abs_factor = context.get("max_abs_factor")
        if len(self.root) < min_count:
            raise ValueError(f"{label} must contain at least {min_count} interventions")
        if max_count is not None and len(self.root) > int(max_count):
            raise ValueError(
                f"{label} must contain at most {int(max_count)} interventions"
            )
        for *coordinate, factor in self.root:
            if abs(factor) < min_factor:
                raise ValueError(
                    f"{label} factor at {tuple(coordinate)} must have magnitude at "
                    f"least {min_factor:g}"
                )
            if max_abs_factor is not None and abs(factor) > float(max_abs_factor):
                raise ValueError(
                    f"{label} factor at {tuple(coordinate)} must have magnitude at "
                    f"most {float(max_abs_factor):g}"
                )
        return self


def compose_multipliers(
    corruption: Sequence[Intervention], restorations: Sequence[Intervention]
) -> dict[CoordinateTuple, float]:
    """Multiply corruption and restoration factors at each coordinate."""

    factors = {item[:3]: item[3] for item in corruption}
    for layer, feature, position, weight in restorations:
        coordinate = (layer, feature, position)
        factors[coordinate] = factors.get(coordinate, 1.0) * weight
    return factors
