import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from .experiments import DEFAULT_BUDGET, AblationTools, CircuitState
from .graphs import (
    Graph,
    graph_to_json,
    load_graph_json,
    render_graph,
    slim_graph,
)
from .rubric import (
    MIN_GROUP_SIZE,
    REWARD_VERSION,
    InterventionScoring,
    parse_proposal,
)

DATA_DIR = Path(__file__).parent / "data"

SYSTEM_PROMPT = f"""\
You are analyzing an attribution graph of Qwen3-4B built with per-layer
transcoders. Group FEATURE nodes by concept and causal role, then connect the
groups into a directed circuit.

You may call circuit_run_ablation before answering. It clamps selected FEATURE
nodes to zero and reports changes in output probability and requested probe-group
activations. Use it to test uncertain causal links.

Your final proposal is scored against a fixed hidden causal-response bank. Members
of one supernode must predict one another's held-out intervention responses, and
the DAG must predict which destination groups and output change. Omitted responses
are predicted as zero. Every submitted member and edge also costs literal
description bits, so include structure only when it improves held-out prediction.

Rules — violations are rejected, not repaired:
- 1-8 disjoint supernodes, at least {MIN_GROUP_SIZE} FEATURE ids each, FEATURE ids only.
- Edges form a duplicate-free DAG; "output" is the reserved terminal.
- Edges mean reachability: A -> B -> output predicts that intervening on A
  changes B and the output with a shared signed response pattern.
- Names are annotations; causal membership and topology determine the score.

Finish with a short analysis and exactly one JSON code block:
```json
{{
  "supernodes": [{{"name": "<concept>", "nodes": ["<node_id>", ...]}}, ...],
  "edges": [["<source>", "<target or output>"], ...]
}}
```"""


class CircuitTaskData(vf.TaskData):
    family: str
    subject_prompt: str
    graph: dict
    experiment_budget: int = DEFAULT_BUDGET
    reward_mode: Literal["train", "eval"] = "eval"
    fault_rate: float = 0.0
    response_bank: dict | None = None


_SCORER = InterventionScoring()


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
    tools = (AblationTools,)

    @vf.metric
    async def experiments_used(self, trace: vf.Trace) -> float:
        return float(trace.state.experiments_used)

    @vf.reward
    async def circuit_faithfulness(self, trace: vf.Trace) -> float:
        # Infra failures deliberately propagate: verifiers records a task error
        # the trainer can drop, where a 0.0 would corrupt the group's advantages.
        if fails_injected_fault(trace.id, self.data.fault_rate):
            raise RuntimeError("injected scoring failure (fault_rate canary)")
        proposal = parse_proposal(trace.last_reply)
        if proposal is None:
            result = {
                "reward": 0.0,
                "net_reward": 0.0,
                "gated": "missing proposal JSON",
                "parse": False,
                "reward_version": REWARD_VERSION,
            }
        else:
            result = await _SCORER.score(
                self.data.subject_prompt,
                load_graph_json(self.data.graph),
                proposal,
                response_bank=self.data.response_bank,
            )
        trace.info["reward_components"] = result
        # V4 deliberately has no parse/membership/topology side reward.  Train
        # and eval retain separate configuration names for compatibility, but
        # both return the same exponential transform of the MDL objective.
        return result["reward"]


class CircuitTasksetConfig(vf.TasksetConfig):
    dataset_file: str = "dataset.jsonl"
    max_features: int = Field(60, ge=2, le=4096)
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
                graph = slim_graph(
                    load_graph_json(row["graph"]), self.config.max_features
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
            f"Expected continuation: {row['target']!r}\n\n"
            f"{render_graph(graph)}\n\n"
            f"Experiment budget: {budget}. "
            "Submit the final circuit when ready."
        )
        data = CircuitTaskData(
            idx=idx,
            name=f"{row['family']}-{idx}-b{budget}",
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            family=row["family"],
            subject_prompt=row["prompt"],
            graph=graph_to_json(graph),
            experiment_budget=budget,
            reward_mode=self.config.reward_mode,
            fault_rate=self.config.fault_rate,
            response_bank=row.get("response_bank"),
        )
        return CircuitTask(data, self.config.task)
