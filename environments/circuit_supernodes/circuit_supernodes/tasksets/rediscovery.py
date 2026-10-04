"""Promptless goal-supernode discovery over hidden prompt cohorts."""

from __future__ import annotations

__all__ = ["DiscoveryTaskset"]

from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from ..models import CircuitDatasetRow, DatasetExample
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows, resolve_dataset
from .shared import CausalCircuitTask, CircuitTaskData, SubjectTasksetConfig

DATA_DIR = Path(__file__).parents[1] / "data" / "supernode_rediscovery"


SYSTEM_PROMPT = """\
Discover the requested goal supernode by attributing arbitrary prompts and
transferring selected features to different prompts in intervention experiments.

Submit named supernodes and a multiplicative intervention for every
member. Find features whose negative intervention removes the goal answer while
remaining specific to that semantic mechanism rather than generic language
behavior.

Feature identity is the stable `layer_feature` pair; each feature is applied at
its highest-ranked position in each attributed graph.

Attribute returns a graph ID and summary. Use query_graph to inspect ranked
nodes, edges, and tokens in bounded pages; follow next_offset with the same
filters. Each new attribution deletes the previous graph. Keep analysis concise
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


def cohort_prompt(goal: str) -> str:
    return (
        f"Goal supernode: {goal!r}\n"
        "Attribute prompts and run interventions with the circuit tools, then submit "
        "the weighted semantic supernodes whose negation disrupts that mechanism "
        "across varied outputs."
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


class DiscoveryTasksetConfig(SubjectTasksetConfig):
    dataset_file: str = "promptless-supernodes.jsonl.gz"
    min_cohort_success: float = Field(1.0, ge=0.0, le=1.0)
    max_control_activation: float = Field(0.0, ge=0.0, le=1.0)


# ty currently treats Verifiers' specialized Task as incompatible with its bare,
# invariant TaskT bound; the runtime and Verifiers' own generic resolver support it.
class DiscoveryTaskset(vf.Taskset[DiscoveryTask, DiscoveryTasksetConfig]):
    def load(self) -> Iterator[DiscoveryTask]:
        source = resolve_dataset(DATA_DIR, self.config.dataset_file)
        rows = load_jsonl_rows(source, CircuitDatasetRow)
        for index, row in enumerate(rows):
            goal = row.goal
            if goal is None:
                raise ValueError("rediscovery rows require a goal")
            data = CircuitTaskData(
                idx=index,
                name=f"promptless-{row.family}-{index}",
                prompt=cohort_prompt(goal),
                system_prompt=SYSTEM_PROMPT,
                family=row.family,
                subject_config=self.config.subject,
                goal=goal,
                example_prompts=row.example_prompts,
                heldout_prompts=row.heldout_prompts,
                control_prompts=row.control_prompts,
                min_visible_success=self.config.min_cohort_success,
                max_control_failure=self.config.max_control_activation,
            )
            yield DiscoveryTask(data, self.config.task)
