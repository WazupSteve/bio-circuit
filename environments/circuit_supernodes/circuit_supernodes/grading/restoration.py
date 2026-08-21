"""Validation and output-based scoring for persistent feature corruption."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    FiniteFloat,
    ValidationInfo,
    model_validator,
)

from ..models import DatasetExample, StrictModel
from ..runtime.graphs import resolve_interventions
from ..runtime.modal import ModalSubject, retry_modal_call

MODAL_APP = "circuit-supernodes-persistent-restoration"
MODAL_CLS = "RestorationAttributor"


class RestorationReplayResult(StrictModel):
    model_config = ConfigDict(extra="allow")
    target_token: str | None = None
    standard_target_is_top: Literal[True]
    corruption_changed_top_token: Literal[True]
    restored_target_is_top: bool
    restored_standard_kl: Annotated[FiniteFloat, Field(ge=-1e-6)] | None = None

    @model_validator(mode="after")
    def valid_replay(self, info: ValidationInfo) -> Self:
        expected_target = (info.context or {}).get("expected_target")
        if expected_target is not None and self.target_token != expected_target:
            raise ValueError("restoration result has the wrong target")
        if self.restored_target_is_top and self.restored_standard_kl is None:
            raise ValueError("restored result requires a numeric clean KL")
        return self


def restoration_similarity(result: object, expected_target: str | None = None) -> float:
    """Diagnostic clean-distribution similarity; never a reward function."""

    data = (
        result.model_dump() if isinstance(result, RestorationReplayResult) else result
    )
    parsed = RestorationReplayResult.model_validate(
        data, context={"expected_target": expected_target}
    )
    if not parsed.restored_target_is_top:
        return 0.0
    divergence = parsed.restored_standard_kl
    assert divergence is not None
    return math.exp(-max(0.0, divergence))


def cohort_clean_similarity(results: Sequence[object]) -> float:
    if not results:
        raise ValueError("restoration results must not be empty")
    return math.fsum(restoration_similarity(result) for result in results) / len(
        results
    )


class RestorationMetricEvaluator:
    """Compute restoration diagnostics without participating in reward grading."""

    def __init__(self, subject: ModalSubject) -> None:
        self.subject = subject

    @retry_modal_call
    async def _restore_output(
        self,
        example: DatasetExample,
        corruption: list[tuple[int, int, int, float]],
        restorations: list[tuple[int, int, int, float]],
    ) -> dict[str, Any]:
        return await self.subject.call(
            "restore_output",
            example.prompt,
            corruption,
            restorations,
            example.target_token,
        )

    async def score_cohort(
        self,
        examples: Sequence[DatasetExample],
        interventions: Sequence[tuple[str, float]],
    ) -> list[dict[str, Any]]:
        async def score_one(example: DatasetExample) -> dict[str, Any]:
            hidden = example.corruption
            hidden_positions = {
                f"{layer}_{feature}": (layer, feature, position)
                for layer, feature, position, _ in hidden
            }
            restorations = []
            unresolved = []
            for node, factor in interventions:
                if coordinate := hidden_positions.get(node):
                    restorations.append((*coordinate, factor))
                else:
                    unresolved.append((node, factor))
            restorations.extend(resolve_interventions(example.graph, unresolved))
            raw = await self._restore_output(
                example, list(hidden), sorted(restorations)
            )
            return RestorationReplayResult.model_validate(
                raw, context={"expected_target": example.target_token}
            ).model_dump()

        return list(await asyncio.gather(*(score_one(example) for example in examples)))
