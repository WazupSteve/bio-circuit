import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from .experiments import DEFAULT_BUDGET, CircuitState, RestorationTools
from .graphs import (
    Graph,
    graph_to_json,
    load_graph_json,
    render_graph,
    slim_graph,
)
from .interventions import CORRUPTION_VERSION, MIN_CORRUPTION_NODES
from .restoration import (
    REWARD_VERSION,
    RestorationScoring,
    parse_intervention_graph,
    validate_corruption,
)

DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = """\
You are given an attribution graph for a model response and the original next
token that must be recovered. Construct a weighted intervention subgraph that
makes that token the model's top next-token output. You may call
circuit_run_restoration to test candidate weights.

An intervention weight is a multiplicative factor on that FEATURE's activation.
Weight 1 is a no-op, 0 ablates it, and other values scale or reverse it. Omitted
features are unchanged. The graph is presented in the same format as an ordinary
circuit-discovery task; infer the useful intervention from its nodes and edges.

The score is zero unless the declared original token becomes top-1. Once it is
top-1, the score is proportional to similarity with the original clean
next-token distribution.

Use only displayed FEATURE ids, do not repeat ids, use at most 60 nodes, and use
finite weights in [-8, 8]. Your final submission must be exactly one JSON code
block and nothing else:
```json
{
  "nodes": [
    {"id": "<feature_id>", "intervention_weight": 1.0}
  ]
}
```"""


class CircuitTaskData(vf.TaskData):
    family: str
    subject_prompt: str
    graph: dict
    target_token: str
    corruption: list[list[int | float]]
    experiment_budget: int = DEFAULT_BUDGET
    reward_mode: Literal["train", "eval"] = "eval"
    fault_rate: float = 0.0


_SCORER = RestorationScoring()


def fails_injected_fault(rollout_id: str, rate: float) -> bool:
    """Canary instrument (``fault_rate = 0`` everywhere else).

    Whether a scoring exception makes the trainer drop the rollout or silently
    contribute a zero is undocumented and decides whether this reward is safe to
    train on at all, so the run that answers it needs a way to fail on demand.
    Deterministic in the rollout id, so a canary is reproducible.
    """
    if rate <= 0.0:
        return False
    draw = hashlib.blake2b(rollout_id.encode(), digest_size=4).digest()
    return int.from_bytes(draw, "big") / 2**32 < rate


class CircuitTask(vf.Task[CircuitTaskData, CircuitState]):
    tools = (RestorationTools,)

    @vf.metric
    async def experiments_used(self, trace: vf.Trace) -> float:
        return float(trace.state.experiments_used)

    @vf.reward
    async def output_restoration(self, trace: vf.Trace) -> float:
        # Infra failures deliberately propagate: verifiers records a task error
        # the trainer can drop, where a 0.0 would corrupt the group's advantages.
        if fails_injected_fault(trace.id, self.data.fault_rate):
            raise RuntimeError("injected scoring failure (fault_rate canary)")
        proposal = parse_intervention_graph(trace.last_reply)
        if proposal is None:
            result = {
                "reward": 0.0,
                "gated": "missing weighted intervention graph JSON",
                "parse": False,
                "reward_version": REWARD_VERSION,
            }
        else:
            result = await _SCORER.score(
                self.data.subject_prompt,
                load_graph_json(self.data.graph),
                self.data.corruption,
                self.data.target_token,
                proposal,
            )
        trace.info["reward_components"] = result
        return result["reward"]


class CircuitTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "dataset.jsonl"
    max_features: int = Field(60, ge=MIN_CORRUPTION_NODES, le=4096)
    experiment_budget: int = Field(DEFAULT_BUDGET, ge=0)
    experiment_budgets: list[int] | None = None
    """Emit one task per budget from every graph, for the budget-curve arms.

    Expanding at load time rather than in the file keeps one copy of each graph
    on disk — the graph is by far the largest thing in a row — and keeps a GRPO
    group homogeneous, since every rollout of a task shares its budget.
    """
    reward_mode: Literal["train", "eval"] = "eval"
    fault_rate: float = Field(0.0, ge=0.0, le=1.0)


class CircuitTaskset(vf.Taskset[CircuitTask, CircuitTasksetConfig]):
    def load(self) -> Iterator[CircuitTask]:
        path = DATA_DIR / self.config.dataset_file
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run `uv run python scripts/build_dataset.py`."
            )

        idx = 0
        with path.open() as rows:
            for line in rows:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("corruption") or not row.get("target_token"):
                    raise ValueError(
                        f"{path} contains a legacy clean row; rebuild it with "
                        "scripts/build_dataset.py to create persistent corruptions"
                    )
                if row.get("corruption_version") != CORRUPTION_VERSION:
                    raise ValueError(
                        f"{path} contains unsupported corruption version "
                        f"{row.get('corruption_version')!r}; rebuild the dataset"
                    )
                corruption = validate_corruption(row["corruption"])
                row["corruption"] = corruption
                graph = slim_graph(
                    load_graph_json(row["graph"]),
                    self.config.max_features,
                    {
                        f"{int(layer)}_{int(feature)}_{int(position)}"
                        for layer, feature, position, _ in corruption
                    },
                )
                budgets = self.config.experiment_budgets or [
                    int(row.get("experiment_budget", self.config.experiment_budget))
                ]
                for budget in budgets:
                    yield self._task(idx, row, graph, budget)
                    idx += 1

    def _task(self, idx: int, row: dict, graph: Graph, budget: int) -> CircuitTask:
        prompt = (
            f"Subject prompt: {row['prompt']!r}\n"
            f"Original next token to restore: {row['target_token']!r}\n\n"
            f"{render_graph(graph)}\n\n"
            f"Experiment budget: {budget}. "
            "Submit the final weighted intervention graph when ready."
        )
        data = CircuitTaskData(
            idx=idx,
            name=f"{row['family']}-{idx}-b{budget}",
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            family=row["family"],
            subject_prompt=row["prompt"],
            graph=graph_to_json(graph),
            target_token=row["target_token"],
            corruption=row["corruption"],
            experiment_budget=budget,
            reward_mode=self.config.reward_mode,
            fault_rate=self.config.fault_rate,
        )
        return CircuitTask(data, self.config.task)
