"""Promptless goal-supernode discovery over hidden prompt cohorts."""

from __future__ import annotations

__all__ = ["DiscoveryTaskset"]

from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from ..models import CircuitDatasetRow, DatasetExample
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows
from .shared import CausalCircuitTask, CircuitTaskData

DATA_DIR = Path(__file__).parents[1] / "data" / "supernode_rediscovery"
DEFAULT_MODAL_APP = "circuit-supernodes-gemma-attribution"
DEFAULT_MODAL_CLS = "GemmaAttributor"


SYSTEM_PROMPT = """\
Discover the requested goal supernode by attributing arbitrary prompts and
transferring selected features to different prompts in intervention experiments.

Submit named supernodes and a nonzero multiplicative intervention for every
member. Find features whose negative intervention removes the goal answer while
remaining specific to that semantic mechanism rather than generic language
behavior.

Feature identity is the stable `layer_feature` pair; each feature is applied at
its highest-ranked position in each attributed graph. Keep analysis concise
and reserve enough output budget to emit the required JSON. If the remaining
budget is uncertain, submit the JSON immediately. Return one JSON object:
```json
{
  "supernodes": [
    {
      "name": "goal",
      "nodes": ["<layer>_<feature>", "<layer>_<feature>"],
      "intervention_weight": 1.0
    }
  ]
}
```"""


def cohort_prompt(goal: str, desired_answer: str) -> str:
    return (
        f"Goal supernode: {goal!r}\n"
        f"Desired answer: {desired_answer!r}\n"
        "Attribute prompts and run interventions with the circuit tools, then submit "
        "the "
        "weighted semantic supernodes whose negation removes the goal answer."
    )


class DiscoveryTask(CausalCircuitTask):
    tools = (CircuitExperimentTools,)

    def grading_cohorts(
        self,
    ) -> tuple[list[DatasetExample], list[DatasetExample], list[DatasetExample]]:
        return self.data.heldout_prompts, [], self.data.control_prompts

    @vf.metric
    async def cohort_metrics(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> dict[str, float]:
        result = await self.score_circuit(trace)
        return {
            "cohort_success_rate": float(result.get("visible_success_rate", 0.0)),
            "control_failure_rate": float(result.get("control_failure_rate", 1.0)),
            "cohort_pass": float(bool(result.get("visible_pass"))),
            "controls_pass": float(bool(result.get("controls_pass"))),
        }

    @vf.reward
    async def goal_supernode_intervention(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> float:
        return float((await self.score_circuit(trace))["reward"])


class DiscoveryTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "promptless-supernodes.jsonl"
    min_cohort_success: float = Field(1.0, ge=0.0, le=1.0)
    max_control_activation: float = Field(0.0, ge=0.0, le=1.0)
    modal_app: str = DEFAULT_MODAL_APP
    modal_cls: str = DEFAULT_MODAL_CLS


# ty currently treats Verifiers' specialized Task as incompatible with its bare,
# invariant TaskT bound; the runtime and Verifiers' own generic resolver support it.
class DiscoveryTaskset(vf.Taskset[DiscoveryTask, DiscoveryTasksetConfig]):
    def load(self) -> Iterator[DiscoveryTask]:
        source = DATA_DIR / self.config.dataset_file
        if not source.exists():
            raise FileNotFoundError(
                f"{source} missing; run scripts/build_promptless_dataset.py."
            )
        rows = load_jsonl_rows(source, CircuitDatasetRow)
        for index, row in enumerate(rows):
            goal = row.goal
            desired_answer = row.desired_answer
            if goal is None or desired_answer is None:
                raise ValueError("rediscovery rows require goal and desired_answer")
            data = CircuitTaskData(
                idx=index,
                name=f"promptless-{row.family}-{index}",
                prompt=cohort_prompt(goal, desired_answer),
                system_prompt=SYSTEM_PROMPT,
                family=row.family,
                goal=goal,
                desired_answer=desired_answer,
                example_prompts=row.example_prompts,
                heldout_prompts=row.heldout_prompts,
                control_prompts=row.control_prompts,
                min_visible_success=self.config.min_cohort_success,
                max_control_failure=self.config.max_control_activation,
                modal_app=self.config.modal_app,
                modal_cls=self.config.modal_cls,
            )
            yield DiscoveryTask(data, self.config.task)
