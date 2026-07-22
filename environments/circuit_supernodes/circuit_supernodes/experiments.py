import verifiers.v1 as vf

from .graphs import Graph, feature_coords, load_graph_json
from .rubric import MODAL_APP, MODAL_CLS, relative_drop

MAX_EXPERIMENTS = 7
MAX_ABLATED_FEATURES = 12
MAX_PROBE_GROUPS = 8
MAX_PROBE_FEATURES = 60


class CircuitState(vf.State):
    experiments_used: int = 0


class AblationTools(vf.Toolset[vf.ToolsetConfig, CircuitState]):
    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._attributor = None
        self._graph: Graph | None = None
        self._prompt = self._target = ""

    async def setup_task(self, task) -> None:
        self._graph = load_graph_json(task.graph)
        self._prompt = task.subject_prompt
        self._target = self._graph.top_outputs[0][0] if self._graph.top_outputs else ""

    @property
    def remaining(self) -> int:
        return max(0, MAX_EXPERIMENTS - self.state.experiments_used)

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
        return await self._attributor.intervene.remote.aio(
            self._prompt, groups, self._target, None, [0]
        )

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

        self.state.experiments_used += 1
        try:
            groups = [feature_coords(ablated)] + [
                feature_coords(ids) for ids in probes.values()
            ]
            result = await self._remote(groups)
        except Exception as exc:
            return self._fail(f"ablation failed: {type(exc).__name__}: {exc}")

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
