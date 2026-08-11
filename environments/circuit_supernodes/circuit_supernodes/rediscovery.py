"""Active supernode discovery with agent-authored subject-model prompts."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .graphs import Graph, load_graph_json, render_graph, slim_graph

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PROMPT_BUDGET = 5
DEFAULT_REPLACEMENT_MAX_TV = 0.05
DEFAULT_REPLACEMENT_REWARD_HALF_LIFE = 0.05
MAX_PROMPT_CHARS = 4096
RETRY_DELAYS = (1.0, 4.0, 15.0)
DEFAULT_MODAL_APP = "circuit-supernodes-gemma-attribution"
DEFAULT_MODAL_CLS = "GemmaAttributor"

SYSTEM_PROMPT = """\
Discover the requested curated supernode in a transcoder-equipped subject
model. You are not given a subject prompt or an attribution graph. Instead, use
circuit_run_prompt to send your own diagnostic prompts to the subject model and
inspect the resulting label-free circuit-tracer graphs.

Feature identity is the stable `layer_feature` pair. Tool results display node
ids as `layer_feature_position`; the final position is specific to the prompt and
is ignored by the grader. Design contrasting prompts that make the requested
concept salient, look for recurring layer-feature pairs, and return those pairs.

Return exactly one JSON code block:
```json
{"nodes": ["<layer>_<feature>", "<layer>_<feature>"]}
```
"""


class DiscoveryState(vf.State):
    prompts_used: int = 0


class DiscoveryTaskData(vf.TaskData):
    target_concept: str
    reference_nodes: list[str]
    grading_prompt: str
    grading_target_token: str
    replacement_max_tv: float = DEFAULT_REPLACEMENT_MAX_TV
    replacement_reward_half_life: float = DEFAULT_REPLACEMENT_REWARD_HALF_LIFE
    prompt_budget: int = DEFAULT_PROMPT_BUDGET
    max_features: int = 150
    modal_app: str = DEFAULT_MODAL_APP
    modal_cls: str = DEFAULT_MODAL_CLS


def parse_nodes(text: str | None) -> list[str] | None:
    """Read the last JSON object containing a node list."""

    if not text:
        return None
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            objects.append(data)
    for data in reversed(objects):
        if not isinstance(data.get("nodes"), list):
            continue
        nodes = data["nodes"]
        if all(isinstance(node, str) and node.strip() for node in nodes):
            return [node.strip() for node in nodes]
    return None


def feature_key(node_id: str) -> str | None:
    """Normalize `layer_feature[_position]` to stable cross-prompt identity."""

    parts = node_id.strip().split("_")
    if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
        return None
    return "_".join(parts[:2])


def membership_score(
    predicted_nodes: list[str] | None, reference_nodes: list[str]
) -> dict[str, float | int | bool]:
    """Cross-prompt feature-identity precision, recall, F1, and exact match."""

    reference = {key for node in reference_nodes if (key := feature_key(node))}
    normalized = [feature_key(node) for node in (predicted_nodes or [])]
    predicted = {key for key in normalized if key is not None}
    invalid_predictions = sum(key is None for key in normalized)
    duplicate_predictions = len([key for key in normalized if key is not None]) - len(
        predicted
    )
    true_positives = len(reference & predicted)
    false_positives = (
        len(predicted - reference) + invalid_predictions + duplicate_predictions
    )
    false_negatives = len(reference - predicted)
    predicted_count = true_positives + false_positives
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    denominator = 2 * true_positives + false_positives + false_negatives
    f1 = 2 * true_positives / denominator if denominator else 0.0
    return {
        "reward": f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": not invalid_predictions
        and not duplicate_predictions
        and predicted == reference,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_count": predicted_count,
        "reference_count": len(reference),
    }


def replacement_coordinates(
    predicted_nodes: list[str] | None, reference_nodes: list[str]
) -> tuple[list[tuple[int, int, int]] | None, str]:
    """Place stable proposal features at the hidden reference positions.

    A proposal that names a reference feature inherits only that feature's exact
    curated positions. A novel feature is tested at every position occupied by
    the reference supernode. This makes an exact stable-id proposal reproduce
    the exact curated intervention while still giving novel features a defined
    cross-prompt position.
    """

    if not predicted_nodes:
        return None, "the final answer does not contain a non-empty node list"

    predicted_keys: list[str] = []
    for node in predicted_nodes:
        key = feature_key(node)
        if key is None:
            return None, f"invalid feature id {node!r}"
        if key in predicted_keys:
            return None, f"duplicate feature identity {key!r}"
        predicted_keys.append(key)

    positions_by_key: dict[str, set[int]] = {}
    reference_positions: set[int] = set()
    for node in reference_nodes:
        parts = node.strip().split("_")
        key = feature_key(node)
        if key is None or len(parts) != 3:
            return None, f"invalid positioned reference feature {node!r}"
        position = int(parts[2])
        positions_by_key.setdefault(key, set()).add(position)
        reference_positions.add(position)
    if not reference_positions:
        return None, "reference supernode has no positions"

    coordinates = {
        (int(layer), int(feature), position)
        for key in predicted_keys
        for layer, feature in [key.split("_")]
        for position in positions_by_key.get(key, reference_positions)
    }
    return sorted(coordinates), ""


def replacement_pass(result: dict[str, Any], max_total_variation: float) -> bool:
    """Require the ablation outputs to retain both token and distribution."""

    try:
        distance = float(result["output_total_variation"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        result.get("valid", False)
        and result.get("top_token_match")
        and math.isfinite(distance)
        and 0.0 <= distance <= max_total_variation
    )


def continuous_replacement_reward(
    result: dict[str, Any], distance_half_life: float
) -> float:
    """Score valid replacements continuously by output-distribution distance.

    Exact functional equivalence has reward 1.0, and every additional
    ``distance_half_life`` of total-variation distance halves the reward. Invalid
    or non-finite backend results retain the 0.0 sentinel.
    """

    if not result.get("valid", False) or not distance_half_life > 0.0:
        return 0.0
    try:
        distance = float(result["output_total_variation"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not math.isfinite(distance) or not 0.0 <= distance <= 1.0:
        return 0.0
    return 2.0 ** (-distance / distance_half_life)


def discovery_graph(graph: Graph, max_features: int) -> Graph:
    """Bound a tool result and remove human-written feature labels."""

    graph = slim_graph(graph, max_features)
    graph.nodes = {
        node_id: replace(node, label="") for node_id, node in graph.nodes.items()
    }
    return graph


async def call_with_retries(
    operation: str, call: Callable[[], Awaitable[Any]]
) -> Any:
    """Retry a remote call and preserve the final exception as the cause."""

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


async def attribute_prompt_with_retries(attributor: Any, prompt: str) -> dict:
    """Run circuit tracer, retrying infrastructure failures before propagating."""

    graph = await call_with_retries(
        "circuit attribution", lambda: attributor.attribute_prompt.remote.aio(prompt)
    )
    if not isinstance(graph, dict):
        raise TypeError("circuit attribution returned a non-object graph")
    return graph


async def replacement_with_retries(attributor: Any, *args: Any) -> dict:
    """Compare hidden reference/proposal interventions with infra retries."""

    result = await call_with_retries(
        "replacement grading",
        lambda: attributor.compare_feature_ablations.remote.aio(*args),
    )
    if not isinstance(result, dict):
        raise TypeError("replacement grading returned a non-object result")
    return result


class CircuitTracerTools(vf.Toolset[vf.ToolsetConfig, DiscoveryState]):
    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._attributor = None
        self._prompt_budget = DEFAULT_PROMPT_BUDGET
        self._max_features = 150
        self._modal_app = DEFAULT_MODAL_APP
        self._modal_cls = DEFAULT_MODAL_CLS

    async def setup_task(self, task: DiscoveryTaskData) -> None:
        self._prompt_budget = task.prompt_budget
        self._max_features = task.max_features
        self._modal_app = task.modal_app
        self._modal_cls = task.modal_cls

    @property
    def remaining(self) -> int:
        return max(0, self._prompt_budget - self.state.prompts_used)

    async def _attribute(self, prompt: str) -> dict:
        if self._attributor is None:
            import modal

            self._attributor = modal.Cls.from_name(self._modal_app, self._modal_cls)()
        return await attribute_prompt_with_retries(self._attributor, prompt)

    @vf.tool(name="run_prompt")
    async def run_prompt(self, prompt: str) -> dict[str, Any]:
        """Run circuit tracer on an agent-authored prompt to the subject model."""

        prompt = str(prompt).strip()
        if not prompt:
            return {
                "error": "prompt must not be empty",
                "prompts_remaining": self.remaining,
            }
        if len(prompt) > MAX_PROMPT_CHARS:
            return {
                "error": f"prompt exceeds the {MAX_PROMPT_CHARS}-character limit",
                "prompts_remaining": self.remaining,
            }
        if not self.remaining:
            return {
                "error": "prompt budget exhausted; submit the final feature set",
                "prompts_remaining": 0,
            }

        raw_graph = await self._attribute(prompt)
        self.state.prompts_used += 1
        graph = discovery_graph(load_graph_json(raw_graph), self._max_features)
        return {
            "graph": render_graph(graph, include_labels=False),
            "feature_nodes": len(graph.feature_nodes),
            "prompts_remaining": self.remaining,
        }


def discovery_prompt(target_concept: str, prompt_budget: int) -> str:
    """Ask for a curated group without supplying a subject prompt or seed graph."""

    return (
        f"Requested supernode description: {target_concept!r}\n"
        f"Subject-prompt budget: {prompt_budget}.\n\n"
        "Create diagnostic prompts, inspect their circuit-tracer graphs, and "
        "identify the recurring features that form the requested supernode."
    )


class DiscoveryTask(vf.Task[DiscoveryTaskData, DiscoveryState]):
    tools = (CircuitTracerTools,)

    @staticmethod
    def _invalid_replacement(reason: str) -> dict[str, Any]:
        return {
            "valid": False,
            "passed": False,
            "reward": 0.0,
            "error": reason,
            "output_total_variation": 1.0,
            "top_token_match": False,
            "reference_effect_total_variation": 0.0,
            "replacement_effect_total_variation": 0.0,
        }

    async def _replacement_result(self, trace: vf.Trace) -> dict[str, Any]:
        cached = trace.info.get("replacement_score")
        if isinstance(cached, dict):
            return cached

        predicted = parse_nodes(trace.last_reply)
        replacement, reason = replacement_coordinates(
            predicted, self.data.reference_nodes
        )
        if replacement is None:
            result = self._invalid_replacement(reason)
            trace.info["replacement_score"] = result
            return result
        if len(replacement) > self.data.max_features:
            result = self._invalid_replacement(
                f"proposal expands to {len(replacement)} positioned features; "
                f"maximum is {self.data.max_features}"
            )
            trace.info["replacement_score"] = result
            return result

        reference = [
            tuple(map(int, node.split("_"))) for node in self.data.reference_nodes
        ]
        import modal

        attributor = modal.Cls.from_name(self.data.modal_app, self.data.modal_cls)()
        result = await replacement_with_retries(
            attributor,
            self.data.grading_prompt,
            reference,
            replacement,
            self.data.grading_target_token,
        )
        result["passed"] = replacement_pass(result, self.data.replacement_max_tv)
        result["reward"] = continuous_replacement_reward(
            result, self.data.replacement_reward_half_life
        )
        result["max_total_variation"] = self.data.replacement_max_tv
        result["reward_half_life"] = self.data.replacement_reward_half_life
        result["predicted_nodes"] = predicted
        result["replacement_coordinates"] = [
            "_".join(map(str, coordinate)) for coordinate in replacement
        ]
        trace.info["replacement_score"] = result
        return result

    @vf.metric
    async def prompts_used(self, trace: vf.Trace) -> float:
        return float(trace.state.prompts_used)

    @vf.metric
    async def precision(self, trace: vf.Trace) -> float:
        result = membership_score(
            parse_nodes(trace.last_reply), self.data.reference_nodes
        )
        return float(result["precision"])

    @vf.metric
    async def recall(self, trace: vf.Trace) -> float:
        result = membership_score(
            parse_nodes(trace.last_reply), self.data.reference_nodes
        )
        return float(result["recall"])

    @vf.metric
    async def exact_match(self, trace: vf.Trace) -> float:
        result = membership_score(
            parse_nodes(trace.last_reply), self.data.reference_nodes
        )
        return float(result["exact_match"])

    @vf.metric
    async def replacement_metrics(self, trace: vf.Trace) -> dict[str, float]:
        result = await self._replacement_result(trace)
        return {
            "replacement_valid": float(result.get("valid", False)),
            "replacement_pass": float(bool(result.get("passed"))),
            "replacement_reward": float(result.get("reward", 0.0)),
            "top_token_match": float(bool(result.get("top_token_match"))),
            "output_total_variation": float(result.get("output_total_variation", 1.0)),
            "reference_effect_total_variation": float(
                result.get("reference_effect_total_variation", 0.0)
            ),
            "replacement_effect_total_variation": float(
                result.get("replacement_effect_total_variation", 0.0)
            ),
        }

    @vf.reward
    async def supernode_replacement(self, trace: vf.Trace) -> float:
        result = await self._replacement_result(trace)
        return float(result["reward"])


class DiscoveryTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "rediscovery.jsonl"
    prompt_budget: int = Field(DEFAULT_PROMPT_BUDGET, ge=1, le=32)
    max_features: int = Field(150, ge=2, le=4096)
    replacement_max_tv: float = Field(DEFAULT_REPLACEMENT_MAX_TV, gt=0.0, le=1.0)
    replacement_reward_half_life: float = Field(
        DEFAULT_REPLACEMENT_REWARD_HALF_LIFE, gt=0.0, le=1.0
    )
    modal_app: str = DEFAULT_MODAL_APP
    modal_cls: str = DEFAULT_MODAL_CLS


class DiscoveryTaskset(vf.Taskset[DiscoveryTask, DiscoveryTasksetConfig]):
    def load(self) -> Iterator[DiscoveryTask]:
        source = DATA_DIR / self.config.dataset_file
        if not source.exists():
            raise FileNotFoundError(
                f"{source} missing; run build_rediscovery_dataset.py first."
            )
        with source.open() as rows:
            for index, line in enumerate(rows):
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                target_concept = str(row["target_concept"]).strip()
                reference_nodes = [str(node) for node in row["reference_nodes"]]
                if not target_concept:
                    raise ValueError(f"rediscovery row {index} has no description")
                if not reference_nodes or len(reference_nodes) != len(
                    set(reference_nodes)
                ):
                    raise ValueError(
                        f"rediscovery row {index} needs non-empty unique references"
                    )
                invalid_references = [
                    node
                    for node in reference_nodes
                    if feature_key(node) is None or len(node.split("_")) != 3
                ]
                if invalid_references:
                    raise ValueError(
                        f"rediscovery row {index} has invalid positioned references: "
                        f"{invalid_references[:4]}"
                    )
                grading = row.get("grading", {})
                grading_prompt = str(grading.get("prompt", "")).strip()
                grading_target_token = str(grading.get("target_token", ""))
                if (not grading_prompt or not grading_target_token) and row.get(
                    "graph"
                ):
                    source_graph = load_graph_json(row["graph"])
                    grading_prompt = (
                        grading_prompt
                        or str(
                            row["graph"].get("metadata", {}).get("prompt", "")
                        ).strip()
                    )
                    if not grading_target_token and source_graph.top_outputs:
                        grading_target_token = source_graph.top_outputs[0][0]
                if not grading_prompt or not grading_target_token.strip():
                    raise ValueError(
                        "rediscovery row needs grading.prompt and grading.target_token"
                    )
                data = DiscoveryTaskData(
                    idx=index,
                    name=f"discover-{target_concept}-{index}",
                    prompt=discovery_prompt(target_concept, self.config.prompt_budget),
                    system_prompt=SYSTEM_PROMPT,
                    target_concept=target_concept,
                    reference_nodes=reference_nodes,
                    grading_prompt=grading_prompt,
                    grading_target_token=grading_target_token,
                    replacement_max_tv=self.config.replacement_max_tv,
                    replacement_reward_half_life=(
                        self.config.replacement_reward_half_life
                    ),
                    prompt_budget=self.config.prompt_budget,
                    max_features=self.config.max_features,
                    modal_app=self.config.modal_app,
                    modal_cls=self.config.modal_cls,
                )
                yield DiscoveryTask(data, self.config.task)


# Compatibility names for direct Python imports. The Verifiers plugin is exported
# separately by the ``supernode_rediscovery`` package.
RediscoveryTask = DiscoveryTask
RediscoveryTaskset = DiscoveryTaskset


if __name__ == "__main__":
    CircuitTracerTools.run()
