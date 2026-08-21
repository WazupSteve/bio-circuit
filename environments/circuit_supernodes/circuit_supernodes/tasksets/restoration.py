"""Cohort task for discovering and reversing persistent feature corruption."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from ..grading.restoration import (
    RestorationMetricEvaluator,
    cohort_clean_kl,
    cohort_clean_similarity,
)
from ..models import CircuitDatasetRow
from ..runtime import circuit_from_response
from ..runtime.circuit_tracer import CircuitExperimentTools
from ..runtime.datasets import load_jsonl_rows
from ..runtime.graphs import weighted_nodes
from ..runtime.modal import subject_from_config
from .shared import CausalCircuitTask, CircuitTaskData, SubjectTasksetConfig

DATA_DIR = Path(__file__).parents[1] / "data" / "output_restoration"

SYSTEM_PROMPT = """\
The subject model has a persistent feature corruption. Attribute arbitrary
prompts to inspect clean feature structure; output logits and tokens are redacted.
Transfer selected features to different prompts in intervention experiments,
whose before/after outputs include the hidden corruption, to identify the mechanism.

Submit stable `layer_feature` ids and multiplicative intervention
factors. Find a shared intervention that reverses the persistent corruption
without capturing unrelated language behavior. Reward requires the original
clean top token and measures `KL(clean || restored)` across the hidden cohort.

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
            replay = await evaluator.score_cohort(
                self.data.heldout_prompts, weighted_nodes(circuit)
            )
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
    dataset_file: str = "suppressed-features.jsonl"


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
                subject_config=self.config.subject,
                example_prompts=row.example_prompts,
                heldout_prompts=row.heldout_prompts,
                control_prompts=row.control_prompts,
                corruption_spec=[
                    (layer, feature, factor)
                    for layer, feature, _, factor in row.heldout_prompts[0].corruption
                ],
            )
            yield CircuitTask(data, self.config.task)
