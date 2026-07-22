import json
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field

from .experiments import MAX_EXPERIMENTS, AblationTools, CircuitState
from .graphs import graph_to_json, load_graph_json, render_graph, slim_graph
from .rubric import InterventionScoring, parse_proposal

DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """\
You are analyzing an attribution graph of Qwen3-4B built with per-layer
transcoders. Group FEATURE nodes by concept and causal role, then connect the
groups into a directed circuit.

You may call circuit_run_ablation before answering. It clamps selected FEATURE
nodes to zero and reports changes in output probability and requested probe-group
activations. Use it to test uncertain causal links.

Your final proposal is scored by independent ablations for causal separation,
feature coverage, and completeness. Use 2-8 disjoint supernodes, FEATURE IDs
only. Every supernode needs an outgoing edge. "output" is the reserved terminal.
Edges mean reachability: A -> B -> output predicts that ablating A reduces B and
the output.

Finish with a short analysis and exactly one JSON code block:
```json
{
  "supernodes": [{"name": "<concept>", "nodes": ["<node_id>", ...]}, ...],
  "edges": [["<source>", "<target or output>"], ...]
}
```"""


class CircuitTaskData(vf.TaskData):
    family: str
    subject_prompt: str
    graph: dict


_SCORER = InterventionScoring()


class CircuitTask(vf.Task[CircuitTaskData, CircuitState]):
    tools = (AblationTools,)

    @vf.metric
    async def experiments_used(self, trace: vf.Trace) -> float:
        return float(trace.state.experiments_used)

    @vf.reward
    async def circuit_faithfulness(self, trace: vf.Trace) -> float:
        proposal = parse_proposal(trace.last_reply)
        if proposal is None:
            trace.info["scoring_error"] = "missing final proposal JSON"
            return 0.0
        try:
            result = await _SCORER.score(
                self.data.subject_prompt,
                load_graph_json(self.data.graph),
                proposal,
            )
        except Exception as exc:
            trace.info["intervention_error"] = f"{type(exc).__name__}: {exc}"
            return 0.0
        trace.info["reward_components"] = result
        return result["reward"]


class CircuitTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "dataset.jsonl"
    max_features: int = Field(60, ge=2, le=4096)


class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = DATA_DIR / self.config.dataset_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run `uv run python scripts/build_dataset.py`."
            )

        with path.open() as rows:
            for idx, line in enumerate(rows):
                if not line.strip():
                    continue
                row = json.loads(line)
                graph = slim_graph(
                    load_graph_json(row["graph"]), self.config.max_features
                )
                prompt = (
                    f"Subject prompt: {row['prompt']!r}\n"
                    f"Expected continuation: {row['target']!r}\n\n"
                    f"{render_graph(graph)}\n\n"
                    f"Experiment budget: {MAX_EXPERIMENTS}. "
                    "Submit the final circuit when ready."
                )
                data = CircuitTaskData(
                    idx=idx,
                    name=f"{row['family']}-{idx}",
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    family=row["family"],
                    subject_prompt=row["prompt"],
                    graph=graph_to_json(graph),
                )
                yield CircuitTask(data, self.config.task)
