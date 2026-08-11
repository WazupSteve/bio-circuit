import verifiers.v1 as vf

from .graphs import Graph, feature_coords, load_graph_json, render_graph, slim_graph
from .rubric import (
    MIN_GROUP_SIZE,
    MODAL_APP,
    MODAL_CLS,
    intervene_with_retries,
    relative_drop,
)

DEFAULT_BUDGET = 7
MAX_ABLATED_FEATURES = 12
MAX_PROBE_GROUPS = 8
MAX_PROBE_FEATURES = 60


class CircuitState(vf.State):
    experiments_used: int = 0


def graph_view(graph: Graph, feature_count: int) -> dict[str, int | str]:
    """Render an exact top-k view of the complete exported graph."""

    available_feature_count = len(graph.feature_nodes)
    if not MIN_GROUP_SIZE <= feature_count <= available_feature_count:
        raise ValueError(
            f"feature_count must be between {MIN_GROUP_SIZE} and "
            f"{available_feature_count}"
        )
    return {
        "returned_feature_count": feature_count,
        "available_feature_count": available_feature_count,
        "graph": render_graph(slim_graph(graph, feature_count)),
    }


class AblationTools(vf.Toolset[vf.ToolsetConfig, CircuitState]):
    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._attributor = None
        self._graph: Graph | None = None
        self._prompt = self._target = ""
        self._budget = DEFAULT_BUDGET

    async def setup_task(self, task) -> None:
        self._graph = load_graph_json(task.graph)
        self._prompt = task.subject_prompt
        self._target = self._graph.top_outputs[0][0] if self._graph.top_outputs else ""
        self._budget = task.experiment_budget

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self.state.experiments_used)

    def _fail(self, message: str) -> dict:
        return {"error": message, "experiments_remaining": self.remaining}

    def _features(self, ids: list[str]) -> list[str]:
        if self._graph is None:
            raise ValueError("tool is not initialized")
        ids = list(dict.fromkeys(map(str, ids)))
        if not ids:
            raise ValueError("provide at least one FEATURE ID")
        invalid = [
            node_id
            for node_id in ids
            if node_id not in self._graph.nodes
            or self._graph.nodes[node_id].kind != "feature"
        ]
        if invalid:
            raise ValueError(f"invalid FEATURE IDs: {invalid}")
        return ids

    async def _remote(self, groups: list) -> dict:
        if self._attributor is None:
            import modal

            self._attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
        return await intervene_with_retries(
            self._attributor, self._prompt, groups, self._target, None, [0]
        )

    @vf.tool(name="view_graph")
    async def view_graph(self, feature_count: int) -> dict:
        """Return the top ``feature_count`` FEATURE nodes and their induced graph."""
        if self._graph is None:
            return self._fail("tool is not initialized")
        try:
            payload = graph_view(self._graph, feature_count)
        except ValueError as exc:
            return self._fail(str(exc))
        return {**payload, "experiments_remaining": self.remaining}

    @vf.tool(name="run_ablation")
    async def run_ablation(
        self,
        feature_ids: list[str],
        probe_groups: dict[str, list[str]] | None = None,
    ) -> dict:
        """Clamp FEATURE IDs and measure output and probe-group activation changes."""
        if not self.remaining:
            return self._fail("experiment budget exhausted; submit the final proposal")
        if not self._target.strip():
            return self._fail("graph has no target token")

        try:
            ablated = self._features(feature_ids)
            probes = {
                name.strip(): self._features(ids)
                for name, ids in (probe_groups or {}).items()
                if name.strip()
            }
            if len(ablated) > MAX_ABLATED_FEATURES:
                raise ValueError(
                    f"at most {MAX_ABLATED_FEATURES} features may be ablated"
                )
            if len(probes) != len(probe_groups or {}):
                raise ValueError("probe names must be non-empty")
            if len(probes) > MAX_PROBE_GROUPS:
                raise ValueError(f"at most {MAX_PROBE_GROUPS} probe groups are allowed")
            if sum(map(len, probes.values())) > MAX_PROBE_FEATURES:
                raise ValueError(
                    f"at most {MAX_PROBE_FEATURES} probe features are allowed"
                )
        except ValueError as exc:
            return self._fail(str(exc))

        # Budget is spent on evidence, not on infra faults: charge after the call
        # returns, and let an exhausted retry chain raise rather than silently
        # burning the agent's experiments.
        groups = [feature_coords(ablated)] + [
            feature_coords(ids) for ids in probes.values()
        ]
        result = await self._remote(groups)
        self.state.experiments_used += 1

        probe_results = {}
        for name, base, changed in zip(
            probes,
            result["base_group_acts"][1:],
            result["ablations"][0]["group_acts"][1:],
            strict=True,
        ):
            probe_results[name] = {
                "baseline": base,
                "ablated": changed,
                "relative_drop": relative_drop(base, changed),
            }
        base = result["base_prob"]
        changed = result["ablations"][0]["target_prob"]
        return {
            "target": {
                "baseline_probability": base,
                "ablated_probability": changed,
                "relative_drop": relative_drop(base, changed),
            },
            "probe_groups": probe_results,
            "experiments_remaining": self.remaining,
        }


if __name__ == "__main__":
    AblationTools.run()
