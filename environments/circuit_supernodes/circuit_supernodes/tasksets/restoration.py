"""Cohort task for discovering and reversing persistent feature corruption."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import TypeAdapter, ValidationError

from ..grading.restoration import (
    RestorationMetricEvaluator,
    cohort_clean_kl,
    cohort_clean_similarity,
)
from ..models import CircuitDatasetRow, Intervention
from ..runtime import circuit_from_response
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows, resolve_dataset
from ..runtime.graphs import weighted_nodes
from ..runtime.modal import subject_from_config
from .shared import (
    CausalCircuitTask,
    CircuitTaskData,
    SubjectTasksetConfig,
    grading_validation_failure,
    is_lazy_dataset,
    row_headers,
    row_section,
)

DATA_DIR = Path(__file__).parents[1] / "data" / "output_restoration"
_EXAMPLE_START = re.compile(rb'\{\s*"prompt"\s*:')
_CORRUPTION_RE = re.compile(rb'"corruption"\s*:\s*(\[[^"{}]*?\]\s*\])')
_INTERVENTIONS = TypeAdapter(list[Intervention])


def first_heldout_corruption(line: bytes) -> list[Intervention]:
    """Read the first held-out corruption without validating cohort graphs."""

    heldout = row_section(line, "heldout_prompts", "control_prompts")
    starts = _EXAMPLE_START.finditer(heldout)
    next(starts, None)
    second = next(starts, None)
    first = heldout if second is None else heldout[: second.start()]
    match = _CORRUPTION_RE.search(first)
    if match is None:
        return []
    return _INTERVENTIONS.validate_python(
        [tuple(item) for item in json.loads(match[1])]
    )


SYSTEM_PROMPT = """\
The subject model has a persistent feature corruption. Attribute arbitrary
prompts to inspect clean feature structure; output logits and tokens are redacted.
Transfer selected features to different prompts in intervention experiments,
whose before/after outputs include the hidden corruption, to identify the mechanism.

Submit stable `layer_feature` ids and multiplicative intervention
factors. Find a shared intervention that reverses the persistent corruption
without capturing unrelated language behavior. Reward requires the original
clean top token and measures `KL(clean || restored)` across the hidden cohort.

Attribute returns a graph ID and summary. Use query_graph to inspect ranked
nodes, edges, and tokens in bounded pages; follow next_offset with the same
filters. Each new attribution deletes the previous graph.

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


def restoration_prompt(family: str) -> str:
    return (
        f"Corrupted semantic family: {family!r}. Attribute prompts, test shared "
        "multipliers, and submit the restoration."
    )


class CircuitTask(CausalCircuitTask):
    tools = (CircuitExperimentTools,)
    score_cache_key = "restoration_score"
    parse_options = {"require_supernodes": False}

    async def score_circuit(self, trace: vf.Trace[CircuitTaskData, vf.State]) -> dict:
        cached = trace.info.get(self.score_cache_key)
        if isinstance(cached, dict):
            return cached
        circuit, reason = circuit_from_response(trace.last_reply)
        if circuit is None:
            result = {
                "reward": 0.0,
                "valid": False,
                "error": reason,
            }
        else:
            evaluator = RestorationMetricEvaluator(
                subject_from_config(self.data.subject_config)
            )
            _, heldout, _ = await asyncio.to_thread(self.grading_cohorts)
            try:
                replay = await evaluator.score_cohort(heldout, weighted_nodes(circuit))
                similarity = cohort_clean_similarity(replay)
                result = {
                    "reward": similarity,
                    "valid": True,
                    "heldout_clean_similarity": similarity,
                    "heldout_clean_kl": cohort_clean_kl(replay),
                    "restored_target_rate": sum(
                        item.get("restored_target_is_top") is True for item in replay
                    )
                    / len(replay),
                    "proposal": weighted_nodes(circuit),
                }
            except ValidationError as exc:
                result = grading_validation_failure(exc)
        trace.info[self.score_cache_key] = result
        return result

    @vf.metric
    async def restoration_metrics(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> dict[str, float]:
        score = await self.score_circuit(trace)
        return {
            "heldout_clean_similarity": float(
                score.get("heldout_clean_similarity", 0.0)
            ),
            "heldout_clean_kl": float(score.get("heldout_clean_kl", 0.0)),
            "restored_target_rate": float(score.get("restored_target_rate", 0.0)),
        }

    @vf.reward
    async def suppressed_feature_restoration(
        self, trace: vf.Trace[CircuitTaskData, vf.State]
    ) -> float:
        return float((await self.score_circuit(trace))["reward"])


class CircuitTasksetConfig(SubjectTasksetConfig):
    dataset_file: str = "suppressed-features.jsonl.gz"


# See the matching note in rediscovery.py about Verifiers' invariant TaskT bound.
class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = resolve_dataset(DATA_DIR, self.config.dataset_file)
        if is_lazy_dataset(path):
            for task_index, family, _, line in row_headers(path):
                yield self.build_task(
                    task_index,
                    family,
                    first_heldout_corruption(line),
                    dataset_path=str(path),
                    dataset_row=task_index,
                )
            return
        rows = load_jsonl_rows(path, CircuitDatasetRow)
        for task_index, row in enumerate(rows):
            yield self.build_task(
                task_index,
                row.family,
                row.heldout_prompts[0].corruption,
                example_prompts=row.example_prompts,
                heldout_prompts=row.heldout_prompts,
                control_prompts=row.control_prompts,
            )

    def build_task(
        self,
        task_index: int,
        family: str,
        corruption: list[Intervention],
        **cohorts: Any,
    ) -> CircuitTask:
        data = CircuitTaskData(
            idx=task_index,
            name=f"suppressed-{family}-{task_index}",
            prompt=restoration_prompt(family),
            system_prompt=SYSTEM_PROMPT,
            family=family,
            subject_config=self.config.subject,
            corruption_spec=[
                (layer, feature, factor) for layer, feature, _, factor in corruption
            ],
            **cohorts,
        )
        return CircuitTask(data, self.config.task)
