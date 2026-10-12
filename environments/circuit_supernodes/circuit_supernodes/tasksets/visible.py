import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from ..models import Circuit, CircuitDatasetRow
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows, resolve_dataset
from .shared import (
    JSON_STRING,
    CausalCircuitTask,
    CircuitTaskData,
    SubjectTasksetConfig,
    is_lazy_dataset,
    row_headers,
    row_section,
)

DATA_DIR = Path(__file__).parents[1] / "data" / "visible_dataset"
_EXAMPLE_RE = re.compile(
    rb'"prompt"\s*:\s*('
    + JSON_STRING
    + rb')\s*,\s*"target_token"\s*:\s*('
    + JSON_STRING
    + rb")"
)


@dataclass(frozen=True)
class _PromptHeader:
    prompt: str
    target_token: str


def _prompt_headers(section: bytes) -> list[_PromptHeader]:
    return [
        _PromptHeader(json.loads(match[1]), json.loads(match[2]))
        for match in _EXAMPLE_RE.finditer(section)
    ]


def _load_headers(
    path: Path,
) -> Iterator[tuple[int, str, str | None, list[_PromptHeader]]]:
    for index, family, goal, line in row_headers(path):
        examples = _prompt_headers(
            row_section(line, "example_prompts", "heldout_prompts")
        )
        if not examples:
            examples = _prompt_headers(
                row_section(line, "heldout_prompts", "control_prompts")
            )[:5]
        yield index, family, goal, examples


SYSTEM_PROMPT = """\
Infer the semantic concept shared by the provided examples, then identify one
supernode of stable FEATURE identities for that concept. You may attribute
arbitrary prompts and transfer a selected feature to a different arbitrary
prompt for an intervention experiment.

Use `layer_feature` identities so a feature denotes the same learned mechanism
regardless of its token position; each feature is applied at its highest-ranked
position in each attributed graph. Find features that are necessary for the
original top token throughout the examples while remaining specific to the
shared semantic concept rather than capturing generic language behavior. The
concept is the full semantic task family, not one answer label.

Attribute returns a graph ID and summary. Use query_graph to inspect ranked
nodes, edges, and tokens in bounded pages; follow next_offset with the same
filters. Each new attribution deletes the previous graph.

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

    def grading_cohorts(self):
        row = self.source_row()
        if row is None:
            return super().grading_cohorts()
        examples = row.example_prompts
        holdouts = row.heldout_prompts
        if not examples and len(holdouts) >= 5:
            examples, holdouts = holdouts[:5], holdouts[5:]
        return examples, holdouts, row.control_prompts

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
    dataset_file: str = "heldout-cohorts.jsonl.gz"
    min_evidence_success: float = Field(1.0, ge=0.0, le=1.0)
    min_holdout_success: float = Field(1.0, ge=0.0, le=1.0)
    max_null_failure: float = Field(0.0, ge=0.0, le=1.0)


# See the matching note in rediscovery.py about Verifiers' invariant TaskT bound.
class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = resolve_dataset(DATA_DIR, self.config.dataset_file)
        if is_lazy_dataset(path):
            for task_index, family, goal, examples in _load_headers(path):
                data = CircuitTaskData(
                    idx=task_index,
                    name=f"circuit-{goal}-{task_index}",
                    prompt=evidence_prompt(examples),
                    system_prompt=SYSTEM_PROMPT,
                    family=family,
                    subject_config=self.config.subject,
                    goal=goal,
                    dataset_path=str(path),
                    dataset_row=task_index,
                    min_visible_success=self.config.min_evidence_success,
                    min_heldout_success=self.config.min_holdout_success,
                    max_control_failure=self.config.max_null_failure,
                )
                yield CircuitTask(data, self.config.task)
            return

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
