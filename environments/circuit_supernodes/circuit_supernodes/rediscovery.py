"""Promptless goal-supernode discovery over hidden prompt cohorts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .cohorts import (
    expand_stable_interventions,
    parse_weighted_nodes,
    success_rate,
    validate_against_graphs,
)
from .graphs import Graph, load_graph_json, render_graph, slim_graph

DATA_DIR = Path(__file__).parent / "data" / "supernode_rediscovery"
DEFAULT_MODAL_APP = "circuit-supernodes-gemma-attribution"
DEFAULT_MODAL_CLS = "GemmaAttributor"
RETRY_DELAYS = (1.0, 4.0, 15.0)

SYSTEM_PROMPT = """\
Discover the requested goal supernode from a cohort of maximal attribution
graphs. The subject prompts are hidden. Call circuit_view_graph with a cohort
index and feature count to inspect an ordinary importance-ranked graph view.

Submit named supernodes and a multiplicative intervention for every member. The
hidden verifier first ablates your proposed nodes on every goal-cohort prompt,
then applies your submitted factors and requires the desired answer to return on
the entire cohort. It also applies the intervention to random control prompts;
none may be driven to the desired answer. This rejects generic target-token
activation circuits.

Feature identity is the stable `layer_feature` pair; each feature is applied at
its highest-ranked position in each stored maximal graph. Return one JSON object:
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


class DiscoveryState(vf.State):
    graph_views: int = 0


class DiscoveryTaskData(vf.TaskData):
    target_concept: str
    desired_answer: str
    desired_target_token: str
    cohort: list[dict]
    controls: list[dict]
    min_cohort_success: float = 1.0
    max_control_activation: float = 0.0
    max_view_features: int = 4096
    modal_app: str = DEFAULT_MODAL_APP
    modal_cls: str = DEFAULT_MODAL_CLS


def promptless_graph(graph: Graph, feature_count: int) -> Graph:
    """Create a label-free, prompt-free view of a stored maximal graph."""

    view = slim_graph(graph, feature_count)
    view.tokens = ["<hidden>"]
    view.top_outputs = []
    view.nodes = {
        node_id: replace(node, token="", label="")
        for node_id, node in view.nodes.items()
    }
    return view


def cohort_prompt(goal: str, desired_answer: str, cohort: int, controls: int) -> str:
    return (
        f"Goal supernode: {goal!r}\n"
        f"Desired answer: {desired_answer!r}\n"
        f"Hidden goal-cohort prompts: {cohort}\n"
        f"Hidden random controls: {controls}\n\n"
        "Inspect stored cohort graphs with circuit_view_graph and submit the "
        "weighted semantic supernodes that causally restore the goal answer."
    )


async def call_with_retries(operation: str, call: Callable[[], Awaitable[Any]]) -> Any:
    attempts = len(RETRY_DELAYS) + 1
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"{operation} failed after {attempts} attempts"
                ) from exc
            await asyncio.sleep(RETRY_DELAYS[attempt])


class CohortGraphTools(vf.Toolset[vf.ToolsetConfig, DiscoveryState]):
    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._graphs: list[Graph] = []
        self._max_view_features = 4096

    async def setup_task(self, task: DiscoveryTaskData) -> None:
        self._graphs = [load_graph_json(example["graph"]) for example in task.cohort]
        self._max_view_features = task.max_view_features

    @vf.tool(name="view_graph")
    async def view_graph(self, cohort_index: int, feature_count: int) -> dict:
        """View a chosen top-k slice without revealing its subject prompt."""

        if not 0 <= cohort_index < len(self._graphs):
            return {"error": f"cohort_index must be in [0, {len(self._graphs) - 1}]"}
        graph = self._graphs[cohort_index]
        available = len(graph.feature_nodes)
        maximum = min(available, self._max_view_features)
        if not 1 <= feature_count <= maximum:
            return {"error": f"feature_count must be between 1 and {maximum}"}
        self.state.graph_views += 1
        view = promptless_graph(graph, feature_count)
        return {
            "cohort_index": cohort_index,
            "returned_feature_count": len(view.feature_nodes),
            "available_feature_count": available,
            "graph": render_graph(view, include_labels=False),
        }


def promptless_score(
    result: dict[str, Any], min_cohort_success: float, max_control_activation: float
) -> dict[str, Any]:
    positives = result.get("cohort")
    controls = result.get("controls")
    if not isinstance(positives, list) or not isinstance(controls, list):
        raise TypeError("cohort verifier returned malformed result lists")
    cohort_rate = success_rate(positives, "shifted_to_target")
    control_rate = success_rate(controls, "intervened_target_is_top")
    cohort_pass = cohort_rate >= min_cohort_success
    controls_pass = control_rate <= max_control_activation
    return {
        **result,
        "cohort_success_rate": cohort_rate,
        "control_activation_rate": control_rate,
        "min_cohort_success": min_cohort_success,
        "max_control_activation": max_control_activation,
        "cohort_pass": cohort_pass,
        "controls_pass": controls_pass,
        "reward": float(cohort_pass and controls_pass),
    }


class DiscoveryTask(vf.Task[DiscoveryTaskData, DiscoveryState]):
    tools = (CohortGraphTools,)

    async def _score(self, trace: vf.Trace) -> dict[str, Any]:
        cached = trace.info.get("cohort_score")
        if isinstance(cached, dict):
            return cached
        proposal, reason = parse_weighted_nodes(
            trace.last_reply, require_supernodes=True
        )
        if proposal is None:
            result = {"reward": 0.0, "valid": False, "error": reason}
            trace.info["cohort_score"] = result
            return result

        cohort_graphs = [load_graph_json(item["graph"]) for item in self.data.cohort]
        valid, reason = validate_against_graphs(cohort_graphs, proposal)
        if not valid:
            result = {"reward": 0.0, "valid": False, "error": reason}
            trace.info["cohort_score"] = result
            return result

        def examples(rows: list[dict], *, controls: bool = False) -> list[dict]:
            return [
                {
                    "prompt": row["prompt"],
                    "target_token": (
                        self.data.desired_target_token
                        if controls
                        else row["target_token"]
                    ),
                    "interventions": expand_stable_interventions(
                        load_graph_json(row["graph"]), proposal
                    ),
                }
                for row in rows
            ]

        import modal

        attributor = modal.Cls.from_name(self.data.modal_app, self.data.modal_cls)()
        raw = await call_with_retries(
            "promptless cohort grading",
            lambda: attributor.evaluate_intervention_cohort.remote.aio(
                examples(self.data.cohort), examples(self.data.controls, controls=True)
            ),
        )
        if not isinstance(raw, dict):
            raise TypeError("cohort verifier returned a non-object")
        result = promptless_score(
            raw, self.data.min_cohort_success, self.data.max_control_activation
        )
        result["valid"] = True
        result["proposal"] = proposal
        trace.info["cohort_score"] = result
        return result

    @vf.metric
    async def graph_views(self, trace: vf.Trace) -> float:
        return float(trace.state.graph_views)

    @vf.metric
    async def cohort_metrics(self, trace: vf.Trace) -> dict[str, float]:
        result = await self._score(trace)
        return {
            "cohort_success_rate": float(result.get("cohort_success_rate", 0.0)),
            "control_activation_rate": float(
                result.get("control_activation_rate", 1.0)
            ),
            "cohort_pass": float(bool(result.get("cohort_pass"))),
            "controls_pass": float(bool(result.get("controls_pass"))),
        }

    @vf.reward
    async def goal_supernode_intervention(self, trace: vf.Trace) -> float:
        return float((await self._score(trace))["reward"])


class DiscoveryTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "promptless-supernodes.jsonl"
    min_cohort_success: float = Field(1.0, ge=0.0, le=1.0)
    max_control_activation: float = Field(0.0, ge=0.0, le=1.0)
    max_view_features: int = Field(4096, ge=1, le=4096)
    modal_app: str = DEFAULT_MODAL_APP
    modal_cls: str = DEFAULT_MODAL_CLS


def _load_examples(row: dict, key: str, index: int) -> list[dict]:
    examples = row.get(key)
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"promptless row {index} needs a non-empty {key} cohort")
    for example_index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"{key} example {example_index} must be an object")
        for field in ("prompt", "target_token", "graph", "sampling"):
            if field not in example:
                raise ValueError(f"{key} example {example_index} lacks {field}")
        graph = load_graph_json(example["graph"])
        if not graph.feature_nodes:
            raise ValueError(f"{key} example {example_index} has no maximal graph")
    return examples


class DiscoveryTaskset(vf.Taskset[DiscoveryTask, DiscoveryTasksetConfig]):
    def load(self) -> Iterator[DiscoveryTask]:
        source = DATA_DIR / self.config.dataset_file
        if not source.exists():
            raise FileNotFoundError(
                f"{source} missing; run scripts/build_promptless_dataset.py."
            )
        with source.open() as rows:
            for index, line in enumerate(rows):
                if not line.strip():
                    continue
                row = json.loads(line)
                goal = str(row.get("target_concept", "")).strip()
                desired_answer = str(row.get("desired_answer", "")).strip()
                desired_target_token = str(row.get("desired_target_token", ""))
                if not goal or not desired_answer or not desired_target_token:
                    raise ValueError(
                        f"promptless row {index} needs goal and desired answer/token"
                    )
                cohort = _load_examples(row, "cohort", index)
                controls = _load_examples(row, "controls", index)
                if any(
                    example["target_token"] != desired_target_token
                    for example in cohort
                ):
                    raise ValueError(
                        f"promptless row {index} cohort target tokens disagree"
                    )
                data = DiscoveryTaskData(
                    idx=index,
                    name=f"promptless-{row.get('family', 'mixed')}-{index}",
                    prompt=cohort_prompt(
                        goal, desired_answer, len(cohort), len(controls)
                    ),
                    system_prompt=SYSTEM_PROMPT,
                    target_concept=goal,
                    desired_answer=desired_answer,
                    desired_target_token=desired_target_token,
                    cohort=cohort,
                    controls=controls,
                    min_cohort_success=self.config.min_cohort_success,
                    max_control_activation=self.config.max_control_activation,
                    max_view_features=self.config.max_view_features,
                    modal_app=self.config.modal_app,
                    modal_cls=self.config.modal_cls,
                )
                yield DiscoveryTask(data, self.config.task)


RediscoveryTask = DiscoveryTask
RediscoveryTaskset = DiscoveryTaskset


if __name__ == "__main__":
    CohortGraphTools.run()
