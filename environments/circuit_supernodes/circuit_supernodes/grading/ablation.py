"""The single reward grader shared by all circuit task families."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import ConfigDict, ValidationInfo, model_validator

from ..models import (
    Circuit,
    DatasetExample,
    StrictModel,
    TracerGraphPayload,
)
from ..runtime.graphs import (
    non_feature_kinds,
    resolve_interventions,
    top_outputs,
    weighted_nodes,
)
from ..runtime.modal import Subject, retry_modal_call


class PositiveAblationResult(StrictModel):
    model_config = ConfigDict(extra="allow")
    expected_token: str
    baseline_expected_is_top: Literal[True]
    changed_top_token: bool


class ControlAblationResult(StrictModel):
    model_config = ConfigDict(extra="allow")
    expected_token: str
    baseline_expected_is_top: Literal[True]
    preserved_top_token: bool


class AblationVerifierResult(StrictModel):
    model_config = ConfigDict(extra="allow")
    positives: list[PositiveAblationResult]
    controls: list[ControlAblationResult]

    @model_validator(mode="after")
    def matches_requested_cohorts(self, info: ValidationInfo) -> Self:
        context = info.context or {}
        positive_tokens = context.get("positive_tokens")
        control_tokens = context.get("control_tokens")
        if positive_tokens is None or control_tokens is None:
            return self
        if len(self.positives) != len(positive_tokens):
            raise ValueError("wrong positive cohort size")
        if len(self.controls) != len(control_tokens):
            raise ValueError("wrong control cohort size")
        if [row.expected_token for row in self.positives] != list(positive_tokens):
            raise ValueError("positive expected tokens changed")
        if [row.expected_token for row in self.controls] != list(control_tokens):
            raise ValueError("control expected tokens changed")
        return self


def negative_intervention(
    graph: TracerGraphPayload, proposal: Sequence[tuple[str, float]]
) -> list[tuple[int, int, int, float]]:
    """Resolve stable ids and negate each submitted multiplicative factor."""

    # A stable feature may occur at a different token position in each prompt,
    # so the proposal is expanded separately for every stored graph.
    return resolve_interventions(
        graph, [(node_id, -factor) for node_id, factor in proposal]
    )


def ablation_payload(
    row: DatasetExample, proposal: Sequence[tuple[str, float]]
) -> dict[str, object]:
    """Build one example's replay request exactly as grading sends it."""

    graph = row.graph
    # The grader tests whether the intervention displaces the graph's original
    # top token, not the dataset's answer string.
    return {
        "prompt": row.prompt,
        "expected_token": top_outputs(graph)[0][0],
        "interventions": negative_intervention(graph, proposal),
        "baseline_interventions": list(row.corruption),
    }


def validate_ablation_result(
    raw: Any,
    positives: Sequence[dict[str, object]],
    controls: Sequence[dict[str, object]],
) -> AblationVerifierResult:
    """Reject replays whose baseline drifted from the requested top tokens."""

    return AblationVerifierResult.model_validate(
        raw,
        context={
            "positive_tokens": [str(row["expected_token"]) for row in positives],
            "control_tokens": [str(row["expected_token"]) for row in controls],
        },
    )


def ablation_score(
    result: dict[str, Any],
    *,
    visible_count: int,
    min_visible_success: float,
    min_holdout_success: float,
    max_control_failure: float,
) -> dict[str, Any]:
    positives = result["positives"]
    controls = result["controls"]
    visible = positives[:visible_count]
    holdout = positives[visible_count:]

    def rate(rows: Sequence[dict[str, Any]], key: str, *, empty: float) -> float:
        return sum(row.get(key) is True for row in rows) / len(rows) if rows else empty

    visible_rate = rate(visible, "changed_top_token", empty=0.0)
    holdout_rate = rate(holdout, "changed_top_token", empty=1.0)
    control_failure_rate = 1.0 - rate(controls, "preserved_top_token", empty=1.0)
    visible_pass = visible_rate >= min_visible_success
    holdout_pass = holdout_rate >= min_holdout_success
    controls_pass = control_failure_rate <= max_control_failure
    return {
        **result,
        "visible_success_rate": visible_rate,
        "holdout_success_rate": holdout_rate,
        "control_failure_rate": control_failure_rate,
        "min_visible_success": min_visible_success,
        "min_holdout_success": min_holdout_success,
        "max_control_failure": max_control_failure,
        "visible_pass": visible_pass,
        "holdout_pass": holdout_pass,
        "controls_pass": controls_pass,
        "reward": float(visible_pass and holdout_pass and controls_pass),
    }


class AblationHoldoutGrader:
    """Grade only whether the negated proposal is necessary and specific."""

    def __init__(self, subject: Subject) -> None:
        self.subject = subject

    async def grade(
        self,
        circuit: Circuit,
        visible: Sequence[DatasetExample],
        holdout: Sequence[DatasetExample],
        control: Sequence[DatasetExample],
        *,
        min_visible_success: float,
        min_holdout_success: float,
        max_control_failure: float,
    ) -> dict[str, Any]:
        proposal = weighted_nodes(circuit)
        raw = await self.score([*visible, *holdout], control, proposal)
        scored = ablation_score(
            raw,
            visible_count=len(visible),
            min_visible_success=min_visible_success,
            min_holdout_success=min_holdout_success,
            max_control_failure=max_control_failure,
        )
        # Logit and error ids parse like FEATURE ids but resolve to nothing, so
        # they are no-ops; report them rather than dropping them silently.
        ignored: dict[str, str] = {}
        for row in visible:
            ignored.update(
                non_feature_kinds(row.graph, [node_id for node_id, _ in proposal])
            )
        return {**scored, "non_feature_node_ids": ignored}

    async def score(
        self,
        positives: Sequence[DatasetExample],
        controls: Sequence[DatasetExample],
        proposal: Sequence[tuple[str, float]],
    ) -> dict[str, Any]:
        positive_payload = [ablation_payload(row, proposal) for row in positives]
        control_payload = [ablation_payload(row, proposal) for row in controls]
        raw = await self._evaluate_cohort(positive_payload, control_payload)
        parsed = validate_ablation_result(raw, positive_payload, control_payload)
        return parsed.model_dump()

    @retry_modal_call
    async def _evaluate_cohort(
        self,
        positives: list[dict[str, object]],
        controls: list[dict[str, object]],
    ) -> Any:
        return await self.subject.call("evaluate_ablation_cohort", positives, controls)
