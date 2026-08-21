"""Cohort task for discovering and reversing persistent feature corruption."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from ..grading.restoration import (
    MODAL_APP,
    MODAL_CLS,
    RestorationMetricEvaluator,
    cohort_clean_similarity,
)
from ..models import CircuitDatasetRow
from ..runtime import circuit_from_response
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows
from ..runtime.graphs import weighted_nodes
from ..runtime.modal import ModalSubject
from .shared import CausalCircuitTask, CircuitTaskData

DATA_DIR = Path(__file__).parents[1] / "data" / "output_restoration"

SYSTEM_PROMPT = """\
The subject model has a persistent feature corruption. Attribute arbitrary
prompts and transfer selected features to different prompts in intervention
experiments to identify that mechanism.

Submit stable `layer_feature` ids and nonzero multiplicative intervention
factors. Find a shared intervention that reverses the persistent corruption
without capturing unrelated language behavior. Clean-distribution similarity
is reported as a diagnostic.

Keep analysis concise and reserve enough output budget to emit the required JSON.
If the remaining budget is uncertain, submit the JSON immediately.

Return one JSON object:
```json
{
  "nodes": [
    {"id": "<layer>_<feature>", "intervention_weight": 0.5}
  ]
}
```"""


_METRICS = RestorationMetricEvaluator(ModalSubject(MODAL_APP, MODAL_CLS))


def restoration_prompt(family: str) -> str:
    return (
        f"Corrupted semantic family: {family!r}. Attribute prompts, test shared "
        "multipliers, and submit the restoration."
    )


class CircuitTask(CausalCircuitTask):
    tools = (CircuitExperimentTools,)
    score_cache_key = "restoration_score"
    parse_options = {"require_supernodes": False}

    @vf.metric
    async def restoration_metrics(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> dict[str, float]:
        score = await self.score_circuit(trace)
        cached = trace.info.get("restoration_metrics")
        if not isinstance(cached, dict):
            circuit, _ = circuit_from_response(trace.last_reply)
            if circuit is None or not score.get("valid"):
                cached = {}
            else:
                heldout = await _METRICS.score_cohort(
                    self.data.heldout_prompts, weighted_nodes(circuit)
                )
                cached = {
                    "discovery_clean_similarity": 0.0,
                    "heldout_clean_similarity": cohort_clean_similarity(heldout),
                }
            trace.info["restoration_metrics"] = cached
        return {
            "discovery_clean_similarity": float(
                cached.get("discovery_clean_similarity", 0.0)
            ),
            "heldout_clean_similarity": float(
                cached.get("heldout_clean_similarity", 0.0)
            ),
            "discovery_ablation_success_rate": float(
                score.get("visible_success_rate", 0.0)
            ),
            "heldout_ablation_success_rate": float(
                score.get("holdout_success_rate", 0.0)
            ),
            "control_failure_rate": float(score.get("control_failure_rate", 1.0)),
        }

    @vf.reward
    async def suppressed_feature_restoration(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> float:
        return float((await self.score_circuit(trace))["reward"])


class CircuitTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "suppressed-features.jsonl"
    min_heldout_success: float = Field(1.0, ge=0.0, le=1.0)
    max_control_failure: float = Field(0.0, ge=0.0, le=1.0)


# See the matching note in rediscovery.py about Verifiers' invariant TaskT bound.
class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = DATA_DIR / self.config.dataset_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run scripts/build_restoration_dataset.py."
            )
        rows = load_jsonl_rows(path, CircuitDatasetRow)
        for task_index, row in enumerate(rows):
            data = CircuitTaskData(
                idx=task_index,
                name=f"suppressed-{row.family}-{task_index}",
                prompt=restoration_prompt(row.family),
                system_prompt=SYSTEM_PROMPT,
                family=row.family,
                example_prompts=row.example_prompts,
                heldout_prompts=row.heldout_prompts,
                control_prompts=row.control_prompts,
                min_heldout_success=self.config.min_heldout_success,
                max_control_failure=self.config.max_control_failure,
                modal_app=MODAL_APP,
                modal_cls=MODAL_CLS,
                corruption_spec=[
                    (layer, feature, factor)
                    for layer, feature, _, factor in row.heldout_prompts[0].corruption
                ],
            )
            yield CircuitTask(data, self.config.task)
