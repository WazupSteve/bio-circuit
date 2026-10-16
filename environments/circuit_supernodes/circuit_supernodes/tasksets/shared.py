"""Task data shared by every circuit-supernode taskset."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field, ValidationError

from ..grading.ablation import AblationHoldoutGrader
from ..models import Circuit, CircuitDatasetRow, DatasetExample
from ..runtime import circuit_from_response
from ..runtime.datasets import jsonl_lines, load_jsonl_row
from ..runtime.graphs import weighted_nodes
from ..runtime.modal import SubjectConfig, subject_from_config

# Larger datasets keep only task-prompt fields in memory and reload the
# graph-heavy row when grading needs it; small fixtures load eagerly.
LAZY_DATASET_BYTES = 64 * 1024 * 1024
JSON_STRING = rb'"(?:\\.|[^"\\])*"'
_FAMILY_RE = re.compile(rb'"family"\s*:\s*(' + JSON_STRING + rb")")
_GOAL_RE = re.compile(rb'"goal"\s*:\s*(null|' + JSON_STRING + rb")")


def _cohort_start(name: str) -> re.Pattern[bytes]:
    return re.compile(rb'"' + name.encode() + rb'"\s*:\s*\[')


def is_lazy_dataset(path: Path) -> bool:
    return path.stat().st_size >= LAZY_DATASET_BYTES


def row_section(line: bytes, name: str, next_name: str) -> bytes:
    """Raw JSON between a row's ``name`` cohort and the following key."""

    start = _cohort_start(name).search(line)
    end = re.compile(rb'\]\s*,\s*"' + next_name.encode() + rb'"\s*:').search(
        line, start.end() if start else 0
    )
    if start is None or end is None:
        raise ValueError(f"row is missing the {name} cohort")
    return line[start.start() : end.start()]


def row_headers(path: Path) -> Iterator[tuple[int, str, str | None, bytes]]:
    """Stream ``(row, family, goal, line)`` without validating cohort graphs."""

    for index, line in jsonl_lines(path):
        # Top-level scalars precede the cohorts; never search inside graphs.
        cohorts = _cohort_start("example_prompts").search(line)
        head = line if cohorts is None else line[: cohorts.start()]
        family = _FAMILY_RE.search(head)
        if family is None:
            raise ValueError(f"{path} row {index + 1} is missing family")
        goal = _GOAL_RE.search(head)
        yield (
            index,
            json.loads(family[1]),
            None if goal is None or goal[1] == b"null" else json.loads(goal[1]),
            line,
        )


def grading_validation_failure(exc: ValidationError) -> dict[str, Any]:
    """Keep invalid replay data out of rewards without losing diagnostics."""

    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    baseline_fields = {
        "baseline_expected_is_top",
        "standard_target_is_top",
        "corruption_changed_top_token",
    }
    baseline_drift = any(
        error["type"] == "literal_error"
        and error["loc"]
        and error["loc"][-1] in baseline_fields
        for error in errors
    )
    return {
        "reward": 0.0,
        "valid": False,
        "failure": "baseline_drift" if baseline_drift else "grading_validation_error",
        "grading_validation_error": True,
        "baseline_drift": baseline_drift,
        "validation_errors": errors,
    }


class SubjectTasksetConfig(vf.TasksetConfig):
    subject: SubjectConfig = Field(default_factory=SubjectConfig)


class CircuitTaskData(vf.TaskData):
    """Universal runtime form of a three-cohort circuit dataset row."""

    family: str
    goal: str | None = None
    # Cohorts contain multi-megabyte attribution graphs needed only while the
    # task is live. Persisting them in every Trace made even startup failures
    # produce enormous traces; resumed runs reload the task data from the
    # configured dataset.
    example_prompts: list[DatasetExample] = Field(default_factory=list, exclude=True)
    heldout_prompts: list[DatasetExample] = Field(default_factory=list, exclude=True)
    control_prompts: list[DatasetExample] = Field(default_factory=list, exclude=True)
    dataset_path: str | None = Field(default=None, exclude=True)
    dataset_row: int | None = Field(default=None, exclude=True)
    min_visible_success: float = Field(0.0, ge=0.0, le=1.0)
    min_heldout_success: float = Field(1.0, ge=0.0, le=1.0)
    max_control_failure: float = Field(0.0, ge=0.0, le=1.0)
    subject_config: SubjectConfig = Field(default_factory=SubjectConfig, exclude=True)
    corruption_spec: list[tuple[int, int, float]] = Field(
        default_factory=list, exclude=True
    )
    # Discovery grades ``a * -w``; restoration composes ``a * c * w``. The
    # intervene tool follows the same convention as the task's grader.
    negate_interventions: bool = True


class CausalCircuitTask(vf.Task[CircuitTaskData, vf.State]):
    """Shared parse, cache, causal replay, and invalid-response behavior."""

    score_cache_key = "cohort_score"
    parse_options: dict[str, Any] = {"require_supernodes": True}

    def source_row(self) -> CircuitDatasetRow | None:
        """Reload a graph-heavy source row only when grading needs it."""

        if self.data.dataset_path is None or self.data.dataset_row is None:
            return None
        return load_jsonl_row(
            Path(self.data.dataset_path), CircuitDatasetRow, self.data.dataset_row
        )

    def grading_cohorts(
        self,
    ) -> tuple[
        Sequence[DatasetExample],
        Sequence[DatasetExample],
        Sequence[DatasetExample],
    ]:
        row = self.source_row()
        if row is not None:
            return row.example_prompts, row.heldout_prompts, row.control_prompts
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

        # Reloading a graph-heavy row takes seconds; keep other rollouts moving.
        visible, holdout, controls = await asyncio.to_thread(self.grading_cohorts)
        grader = AblationHoldoutGrader(subject_from_config(self.data.subject_config))
        try:
            result = await grader.grade(
                circuit,
                visible,
                holdout,
                controls,
                min_visible_success=self.data.min_visible_success,
                min_holdout_success=self.data.min_heldout_success,
                max_control_failure=self.data.max_control_failure,
            )
        except ValidationError as exc:
            result = grading_validation_failure(exc)
        else:
            result["valid"] = True
        result["proposal"] = self.proposal(circuit)
        trace.info[self.score_cache_key] = result
        return result

    @vf.metric
    async def grading_validation_metrics(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> dict[str, float]:
        result = await self.score_circuit(trace)
        return {
            "grading_validation_error": float(
                result.get("grading_validation_error", False)
            ),
            "baseline_drift": float(result.get("baseline_drift", False)),
            "non_feature_node_ids": float(len(result.get("non_feature_node_ids", {}))),
        }
