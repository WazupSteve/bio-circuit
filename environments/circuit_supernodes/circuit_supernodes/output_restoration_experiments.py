"""Selectable maximal-graph views and discovery-cohort restoration trials."""

from __future__ import annotations

import verifiers.v1 as vf

from .cohorts import parse_weighted_nodes, validate_against_graphs
from .graphs import Graph, load_graph_json, render_graph, slim_graph
from .restoration import RestorationScoring

DEFAULT_BUDGET = 7


class CircuitState(vf.State):
    experiments_used: int = 0
    graph_views: int = 0


def graph_view(graph: Graph, feature_count: int) -> dict[str, int | str]:
    available = len(graph.feature_nodes)
    if not 1 <= feature_count <= available:
        raise ValueError(f"feature_count must be between 1 and {available}")
    view = slim_graph(graph, feature_count)
    return {
        "returned_feature_count": len(view.feature_nodes),
        "available_feature_count": available,
        "graph": render_graph(view),
    }


class RestorationTools(vf.Toolset[vf.ToolsetConfig, CircuitState]):
    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._cohort: list[dict] = []
        self._graphs: list[Graph] = []
        self._budget = DEFAULT_BUDGET
        self._max_view_features = 4096
        self._scorer = RestorationScoring()

    async def setup_task(self, task) -> None:
        self._cohort = list(task.cohort)
        self._graphs = [load_graph_json(row["graph"]) for row in self._cohort]
        self._budget = task.experiment_budget
        self._max_view_features = task.max_view_features

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self.state.experiments_used)

    def _fail(self, message: str) -> dict:
        return {"error": message, "experiments_remaining": self.remaining}

    @vf.tool(name="view_graph")
    async def view_graph(self, cohort_index: int, feature_count: int) -> dict:
        """Return a chosen top-k slice of one stored corrupted maximal graph."""

        if not 0 <= cohort_index < len(self._graphs):
            return self._fail(f"cohort_index must be in [0, {len(self._graphs) - 1}]")
        graph = self._graphs[cohort_index]
        maximum = min(len(graph.feature_nodes), self._max_view_features)
        if not 1 <= feature_count <= maximum:
            return self._fail(f"feature_count must be between 1 and {maximum}")
        try:
            payload = graph_view(graph, feature_count)
        except ValueError as exc:
            return self._fail(str(exc))
        self.state.graph_views += 1
        return {
            "cohort_index": cohort_index,
            **payload,
            "experiments_remaining": self.remaining,
        }

    @vf.tool(name="run_restoration")
    async def run_restoration(self, feature_weights: dict[str, float]) -> dict:
        """Test one stable weighted proposal over the discovery cohort."""

        if not self.remaining:
            return self._fail("experiment budget exhausted; submit the final proposal")
        proposal_text = json_text(feature_weights)
        proposal, reason = parse_weighted_nodes(proposal_text)
        if proposal is None:
            return self._fail(reason)
        valid, reason = validate_against_graphs(self._graphs, proposal)
        if not valid:
            return self._fail(reason)
        results = await self._scorer.score_cohort(self._cohort, proposal)
        self.state.experiments_used += 1
        rate = sum(bool(row.get("restored_target_is_top")) for row in results) / len(
            results
        )
        return {
            "discovery_success_rate": rate,
            "restored": [bool(row.get("restored_target_is_top")) for row in results],
            "experiments_remaining": self.remaining,
        }


def json_text(feature_weights: dict[str, float]) -> str:
    import json

    if not isinstance(feature_weights, dict):
        return ""
    return json.dumps(
        {
            "nodes": [
                {"id": node_id, "intervention_weight": weight}
                for node_id, weight in feature_weights.items()
            ]
        }
    )


if __name__ == "__main__":
    RestorationTools.run()
