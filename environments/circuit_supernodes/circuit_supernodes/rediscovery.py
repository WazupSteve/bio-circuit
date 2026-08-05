"""Zero-shot LLM rediscovery of a hidden, pre-identified supernode."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .graphs import Graph, graph_to_json, load_graph_json, render_graph

DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """\
Independently rediscover the requested conceptual supernode in the attribution
graph. You receive no demonstrations, feature labels, experiments, correctness
feedback, or reference membership; infer the group solely from graph structure,
tokens, layers, positions, importance, and edges.

Return exactly one JSON code block with the FEATURE ids you believe belong to the
requested concept:
```json
{"nodes": ["<feature_id>", "<feature_id>"]}
```
"""


class RediscoveryState(vf.State):
    pass


class RediscoveryTaskData(vf.TaskData):
    graph: dict
    target_concept: str
    reference_nodes: list[str]


def parse_nodes(text: str) -> list[str] | None:
    """Read the last JSON object containing a node list."""

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


def membership_score(
    predicted_nodes: list[str] | None, reference_nodes: list[str]
) -> dict[str, float | int | bool]:
    """Set-membership precision, recall, F1, and exact match."""

    reference = set(reference_nodes)
    predicted = set(predicted_nodes or [])
    true_positives = len(reference & predicted)
    false_positives = len(predicted - reference)
    false_negatives = len(reference - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    denominator = 2 * true_positives + false_positives + false_negatives
    f1 = 2 * true_positives / denominator if denominator else 0.0
    return {
        "reward": f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": predicted == reference,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_count": len(predicted),
        "reference_count": len(reference),
    }


def rediscovery_graph(
    graph: Graph, reference_nodes: list[str], max_features: int
) -> Graph:
    """Retain every hidden reference member plus top-ranked distractors."""

    reference = set(reference_nodes)
    invalid = sorted(
        node_id
        for node_id in reference
        if node_id not in graph.nodes or graph.nodes[node_id].kind != "feature"
    )
    if invalid:
        raise ValueError(f"reference contains non-FEATURE ids: {invalid[:4]}")
    if not reference:
        raise ValueError("reference supernode must not be empty")
    if len(reference) > max_features:
        raise ValueError("max_features is smaller than the reference supernode")

    ranked = sorted(graph.feature_nodes, key=lambda node: -node.share)
    fillers = [node.id for node in ranked if node.id not in reference]
    keep = reference | set(fillers[: max_features - len(reference)])
    keep |= {
        node.id
        for node in graph.nodes.values()
        if node.kind in ("embedding", "logit")
    }
    result = Graph(tokens=list(graph.tokens), top_outputs=list(graph.top_outputs))
    result.nodes = {
        node_id: replace(graph.nodes[node_id], label="") for node_id in keep
    }
    result.edges = [
        edge for edge in graph.edges if edge.source in keep and edge.target in keep
    ]
    return result


def rediscovery_prompt(graph: Graph, target_concept: str) -> str:
    """Render candidates without labels, examples, reference ids, or feedback."""

    return (
        f"Requested concept: {target_concept!r}\n\n"
        f"UNLABELED ATTRIBUTION GRAPH\n{render_graph(graph)}\n\n"
        "Select the FEATURE ids that form the requested conceptual supernode."
    )


class RediscoveryTask(vf.Task[RediscoveryTaskData, RediscoveryState]):
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

    @vf.reward
    async def supernode_membership(self, trace: vf.Trace) -> float:
        result = membership_score(
            parse_nodes(trace.last_reply), self.data.reference_nodes
        )
        trace.info["rediscovery_score"] = result
        return float(result["reward"])


class RediscoveryTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "rediscovery.jsonl"
    max_features: int = Field(150, ge=2, le=4096)


class RediscoveryTaskset(
    vf.Taskset[RediscoveryTask, RediscoveryTasksetConfig]
):
    def load(self) -> Iterator[RediscoveryTask]:
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
                graph = rediscovery_graph(
                    load_graph_json(row["graph"]),
                    reference_nodes,
                    self.config.max_features,
                )
                data = RediscoveryTaskData(
                    idx=index,
                    name=f"rediscover-{target_concept}-{index}",
                    prompt=rediscovery_prompt(graph, target_concept),
                    system_prompt=SYSTEM_PROMPT,
                    graph=graph_to_json(graph),
                    target_concept=target_concept,
                    reference_nodes=reference_nodes,
                )
                yield RediscoveryTask(data, self.config.task)


# The package exports one Taskset under the conventional environment name.
CircuitTaskset = RediscoveryTaskset
