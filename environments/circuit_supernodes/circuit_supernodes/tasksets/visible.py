from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from ..models import Circuit, CircuitDatasetRow
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows
from .shared import CausalCircuitTask, CircuitTaskData, SubjectTasksetConfig

DATA_DIR = Path(__file__).parents[1] / "data" / "visible_dataset"


SYSTEM_PROMPT = """\
Infer the semantic concept shared by the provided examples, then identify one
supernode of stable FEATURE identities for that concept. You may attribute
arbitrary prompts and transfer a selected feature to a different arbitrary
prompt for an intervention experiment.

Use `layer_feature` identities so a feature denotes the same learned mechanism
regardless of its token position. Find features that are necessary for the
original top token throughout the examples while remaining specific to the
shared semantic concept rather than capturing generic language behavior. The
concept is the full semantic task family, not one answer label.

Keep analysis concise and reserve enough output budget to emit the required JSON.
If the remaining budget is uncertain, submit the JSON immediately.

Return exactly one JSON object:
```json
{
  "supernodes": [{"name": "<semantic role>", "nodes": ["<layer>_<feature>", "..."]}]
}
```"""


def evidence_prompt(examples) -> str:
    rendered = "\n".join(
        f"  [{index}] prompt={row.prompt!r}; continuation={row.target_token!r}"
        for index, row in enumerate(examples)
    )
    return (
        f"Visible examples ({len(examples)}):\n{rendered}\n\n"
        "Infer their shared semantic concept, run circuit experiments, and submit "
        "one supernode."
    )


class CircuitTask(CausalCircuitTask):
    tools = (CircuitExperimentTools,)
    parse_options = {
        "default_weight": 1.0,
        "exact_groups": 1,
        "require_supernodes": True,
    }

    def proposal(self, circuit: Circuit) -> object:
        return circuit.model_dump()["supernodes"]

    @vf.metric
    async def cohort_metrics(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> dict[str, float]:
        result = await self.score_circuit(trace)
        return {
            "evidence_success_rate": float(result.get("visible_success_rate", 0.0)),
            "holdout_success_rate": float(result.get("holdout_success_rate", 0.0)),
            "null_failure_rate": float(result.get("control_failure_rate", 1.0)),
            "evidence_pass": float(bool(result.get("visible_pass"))),
            "holdout_pass": float(bool(result.get("holdout_pass"))),
            "null_pass": float(bool(result.get("controls_pass"))),
        }

    @vf.reward
    async def circuit_faithfulness(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> float:
        return float((await self.score_circuit(trace))["reward"])


class CircuitTasksetConfig(SubjectTasksetConfig):
    dataset_file: str = "circuit-cohorts.jsonl"
    min_evidence_success: float = Field(1.0, ge=0.0, le=1.0)
    min_holdout_success: float = Field(1.0, ge=0.0, le=1.0)
    max_null_failure: float = Field(0.0, ge=0.0, le=1.0)


# See the matching note in rediscovery.py about Verifiers' invariant TaskT bound.
class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = DATA_DIR / self.config.dataset_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run `uv run python scripts/build_visible_dataset.py`."
            )
        rows = load_jsonl_rows(path, CircuitDatasetRow)
        for task_index, row in enumerate(rows):
            examples = row.example_prompts
            holdouts = row.heldout_prompts
            # Compatibility for datasets built before visible evidence was stored.
            if not examples and len(holdouts) >= 5:
                examples, holdouts = holdouts[:5], holdouts[5:]
            data = CircuitTaskData(
                idx=task_index,
                name=f"circuit-{row.goal}-{task_index}",
                prompt=evidence_prompt(examples),
                system_prompt=SYSTEM_PROMPT,
                family=row.family,
                subject_config=self.config.subject,
                goal=row.goal,
                example_prompts=examples,
                heldout_prompts=holdouts,
                control_prompts=row.control_prompts,
                min_visible_success=self.config.min_evidence_success,
                min_heldout_success=self.config.min_holdout_success,
                max_control_failure=self.config.max_null_failure,
            )
            yield CircuitTask(data, self.config.task)
