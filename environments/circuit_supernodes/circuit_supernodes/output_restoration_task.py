"""Cohort task for discovering and reversing persistent feature corruption."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from .cohorts import (
    parse_weighted_nodes,
    threshold_score,
    validate_against_graphs,
)
from .graphs import load_graph_json
from .interventions import CORRUPTION_VERSION
from .output_restoration_experiments import (
    DEFAULT_BUDGET,
    CircuitState,
    RestorationTools,
)
from .restoration import REWARD_VERSION, RestorationScoring

DATA_DIR = Path(__file__).parent / "data" / "output_restoration"

SYSTEM_PROMPT = """\
The subject model has the same persistent feature corruption across a cohort of
generated prompts. Each stored attribution graph was captured under that
corruption and is otherwise passed through unchanged. Call circuit_view_graph
with a cohort index and feature count to choose an ordinary importance-ranked
view of any maximal graph. Call circuit_run_restoration to test one weighted
proposal across the visible discovery cohort.

Submit stable `layer_feature` ids and multiplicative intervention factors. The
final verifier replays the hidden corruption and requires the proposal to restore
the desired token on more than 80% of discovery prompts and more than 30% of a
separate corrupted held-out cohort. Held-out prompts and graphs are never shown.

Return one JSON object:
```json
{
  "nodes": [
    {"id": "<layer>_<feature>", "intervention_weight": 0.5}
  ]
}
```"""


class CircuitTaskData(vf.TaskData):
    family: str
    cohort: list[dict]
    heldout: list[dict]
    experiment_budget: int = DEFAULT_BUDGET
    discovery_threshold: float = 0.8
    heldout_threshold: float = 0.3
    max_view_features: int = 4096
    reward_mode: Literal["train", "eval"] = "eval"
    fault_rate: float = 0.0


_SCORER = RestorationScoring()


def fails_injected_fault(rollout_id: str, rate: float) -> bool:
    if rate <= 0.0:
        return False
    draw = hashlib.blake2b(rollout_id.encode(), digest_size=4).digest()
    return int.from_bytes(draw, "big") / 2**32 < rate


def restoration_prompt(cohort: list[dict], budget: int) -> str:
    examples = "\n".join(
        f"  [{index}] prompt={row['prompt']!r}; desired token={row['target_token']!r}"
        for index, row in enumerate(cohort)
    )
    return (
        f"Corrupted discovery cohort ({len(cohort)} prompts):\n{examples}\n\n"
        f"Experiment budget: {budget}. Inspect graph subsets, test shared "
        "multipliers, and submit the final restoration."
    )


class CircuitTask(vf.Task[CircuitTaskData, CircuitState]):
    tools = (RestorationTools,)

    async def _score(self, trace: vf.Trace) -> dict:
        cached = trace.info.get("restoration_score")
        if isinstance(cached, dict):
            return cached
        if fails_injected_fault(trace.id, self.data.fault_rate):
            raise RuntimeError("injected scoring failure (fault_rate canary)")
        proposal, reason = parse_weighted_nodes(trace.last_reply)
        if proposal is None:
            result = {
                "reward": 0.0,
                "valid": False,
                "error": reason,
                "reward_version": REWARD_VERSION,
            }
            trace.info["restoration_score"] = result
            return result
        graphs = [load_graph_json(row["graph"]) for row in self.data.cohort]
        valid, reason = validate_against_graphs(graphs, proposal)
        if not valid:
            result = {
                "reward": 0.0,
                "valid": False,
                "error": reason,
                "reward_version": REWARD_VERSION,
            }
            trace.info["restoration_score"] = result
            return result

        discovery = await _SCORER.score_cohort(self.data.cohort, proposal)
        heldout = await _SCORER.score_cohort(self.data.heldout, proposal)
        discovery_rate = sum(
            bool(item.get("restored_target_is_top")) for item in discovery
        ) / len(discovery)
        heldout_rate = sum(
            bool(item.get("restored_target_is_top")) for item in heldout
        ) / len(heldout)
        result = {
            **threshold_score(
                discovery_rate,
                heldout_rate,
                discovery_threshold=self.data.discovery_threshold,
                heldout_threshold=self.data.heldout_threshold,
            ),
            "valid": True,
            "discovery": discovery,
            "heldout": heldout,
            "proposal": proposal,
            "reward_version": REWARD_VERSION,
        }
        trace.info["restoration_score"] = result
        return result

    @vf.metric
    async def experiments_used(self, trace: vf.Trace) -> float:
        return float(trace.state.experiments_used)

    @vf.metric
    async def restoration_metrics(self, trace: vf.Trace) -> dict[str, float]:
        result = await self._score(trace)
        return {
            "discovery_success_rate": float(result.get("discovery_success_rate", 0.0)),
            "heldout_success_rate": float(result.get("heldout_success_rate", 0.0)),
            "discovery_pass": float(bool(result.get("discovery_pass"))),
            "heldout_pass": float(bool(result.get("heldout_pass"))),
        }

    @vf.reward
    async def suppressed_feature_restoration(self, trace: vf.Trace) -> float:
        return float((await self._score(trace))["reward"])


class CircuitTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "suppressed-features.jsonl"
    experiment_budget: int = Field(DEFAULT_BUDGET, ge=0)
    experiment_budgets: list[int] | None = None
    discovery_threshold: float = Field(0.8, ge=0.0, lt=1.0)
    heldout_threshold: float = Field(0.3, ge=0.0, lt=1.0)
    max_view_features: int = Field(4096, ge=1, le=4096)
    reward_mode: Literal["train", "eval"] = "eval"
    fault_rate: float = Field(0.0, ge=0.0, le=1.0)


def _validate_examples(row: dict, key: str, row_index: int) -> list[dict]:
    examples = row.get(key)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"restoration row {row_index} needs non-empty {key}")
    for example_index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"{key} example {example_index} must be an object")
        for field in ("prompt", "target_token", "graph", "corruption", "sampling"):
            if field not in example:
                raise ValueError(f"{key} example {example_index} lacks {field}")
        if example.get("corruption_version") != CORRUPTION_VERSION:
            raise ValueError(f"{key} example {example_index} has wrong corruption")
        graph = load_graph_json(example["graph"])
        exported = {node.id for node in graph.feature_nodes}
        hidden = {
            f"{int(layer)}_{int(feature)}_{int(position)}"
            for layer, feature, position, _ in example["corruption"]
        }
        if not hidden or not hidden.issubset(exported):
            raise ValueError(
                f"{key} example {example_index} corruption is outside maximal graph"
            )
    return examples


class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = DATA_DIR / self.config.dataset_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run scripts/build_restoration_dataset.py."
            )
        task_index = 0
        with path.open() as rows:
            for row_index, line in enumerate(rows):
                if not line.strip():
                    continue
                row = json.loads(line)
                cohort = _validate_examples(row, "cohort", row_index)
                heldout = _validate_examples(row, "heldout", row_index)
                budgets = self.config.experiment_budgets or [
                    int(row.get("experiment_budget", self.config.experiment_budget))
                ]
                for budget in budgets:
                    data = CircuitTaskData(
                        idx=task_index,
                        name=f"suppressed-{row.get('family', 'mixed')}-{task_index}",
                        prompt=restoration_prompt(cohort, budget),
                        system_prompt=SYSTEM_PROMPT,
                        family=str(row.get("family", "mixed")),
                        cohort=cohort,
                        heldout=heldout,
                        experiment_budget=budget,
                        discovery_threshold=self.config.discovery_threshold,
                        heldout_threshold=self.config.heldout_threshold,
                        max_view_features=self.config.max_view_features,
                        reward_mode=self.config.reward_mode,
                        fault_rate=self.config.fault_rate,
                    )
                    yield CircuitTask(data, self.config.task)
                    task_index += 1
