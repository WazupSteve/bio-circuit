"""Task data shared by every circuit-supernode taskset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from ..grading.ablation import AblationHoldoutGrader
from ..models import Circuit, DatasetExample
from ..runtime import circuit_from_response
from ..runtime.graphs import weighted_nodes
from ..runtime.modal import ModalSubject


class CircuitTaskData(vf.TaskData):
    """Universal runtime form of a three-cohort circuit dataset row."""

    family: str
    goal: str | None = None
    example_prompts: list[DatasetExample]
    heldout_prompts: list[DatasetExample]
    control_prompts: list[DatasetExample]
    min_visible_success: float = Field(0.0, ge=0.0, le=1.0)
    min_heldout_success: float = Field(1.0, ge=0.0, le=1.0)
    max_control_failure: float = Field(0.0, ge=0.0, le=1.0)
    modal_app: str
    modal_cls: str
    corruption_spec: list[tuple[int, int, float]] = Field(
        default_factory=list, exclude=True
    )


class CausalCircuitTask(vf.Task[CircuitTaskData, vf.State]):
    """Shared parse, cache, causal replay, and invalid-response behavior."""

    score_cache_key = "cohort_score"
    parse_options: dict[str, Any] = {"require_supernodes": True}

    def grading_cohorts(
        self,
    ) -> tuple[
        Sequence[DatasetExample],
        Sequence[DatasetExample],
        Sequence[DatasetExample],
    ]:
        return (
            self.data.example_prompts,
            self.data.heldout_prompts,
            self.data.control_prompts,
        )

    def proposal(self, circuit: Circuit) -> object:
        return weighted_nodes(circuit)

    async def score_circuit(self, trace: vf.Trace[CircuitTaskData, vf.State]) -> dict:
        cached = trace.info.get(self.score_cache_key)
        if isinstance(cached, dict):
            return cached

        circuit, reason = circuit_from_response(trace.last_reply, **self.parse_options)
        if circuit is None:
            result = {
                "reward": 0.0,
                "valid": False,
                "error": reason,
            }
            trace.info[self.score_cache_key] = result
            return result

        visible, holdout, controls = self.grading_cohorts()
        grader = AblationHoldoutGrader(
            ModalSubject(self.data.modal_app, self.data.modal_cls)
        )
        result = await grader.grade(
            circuit,
            visible,
            holdout,
            controls,
            min_visible_success=self.data.min_visible_success,
            min_holdout_success=self.data.min_heldout_success,
            max_control_failure=self.data.max_control_failure,
        )
        result["valid"] = True
        result["proposal"] = self.proposal(circuit)
        trace.info[self.score_cache_key] = result
        return result
