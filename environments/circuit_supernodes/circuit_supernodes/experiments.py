import verifiers.v1 as vf

from .graphs import Graph, load_graph_json
from .interventions import coordinate_from_feature_id
from .restoration import (
    MODAL_APP,
    MODAL_CLS,
    restore_with_retries,
    validate_intervention_graph,
    validate_replay_invariants,
)

DEFAULT_BUDGET = 7


class CircuitState(vf.State):
    experiments_used: int = 0


class RestorationTools(vf.Toolset[vf.ToolsetConfig, CircuitState]):
    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._attributor = None
        self._graph: Graph | None = None
        self._prompt = self._target = ""
        self._corruption: list[list[int | float]] = []
        self._budget = DEFAULT_BUDGET

    async def setup_task(self, task) -> None:
        self._graph = load_graph_json(task.graph)
        self._prompt = task.subject_prompt
        self._target = task.target_token
        self._corruption = task.corruption
        self._budget = task.experiment_budget

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self.state.experiments_used)

    def _fail(self, message: str) -> dict:
        return {"error": message, "experiments_remaining": self.remaining}

    async def _restore_remote(self, restorations: list[tuple[int, int, int, float]]):
        if self._attributor is None:
            import modal

            self._attributor = modal.Cls.from_name(MODAL_APP, MODAL_CLS)()
        if self._graph is None:
            raise ValueError("tool is not initialized")
        return await restore_with_retries(
            self._attributor,
            self._prompt,
            self._corruption,
            restorations,
            self._target,
        )

    @vf.tool(name="run_restoration")
    async def run_restoration(self, feature_weights: dict[str, float]) -> dict:
        """Test multiplicative weights on displayed FEATURE activations."""

        if not self.remaining:
            return self._fail("experiment budget exhausted; submit the final proposal")
        if not self._target.strip():
            return self._fail("graph has no target token")
        if self._graph is None:
            return self._fail("tool is not initialized")
        if not isinstance(feature_weights, dict):
            return self._fail("feature_weights must be an object")
        proposal = {
            "nodes": [
                {"id": node_id, "intervention_weight": weight}
                for node_id, weight in feature_weights.items()
            ]
        }
        validated, reason = validate_intervention_graph(proposal, self._graph)
        if validated is None:
            return self._fail(reason)
        try:
            restorations = [
                (*coordinate_from_feature_id(node_id), weight)
                for node_id, weight in validated
            ]
        except ValueError as exc:
            return self._fail(str(exc))

        result = await self._restore_remote(restorations)
        validate_replay_invariants(result, self._target)
        self.state.experiments_used += 1
        # Do not reveal clean or hidden-corruption measurements. These are the
        # same observable outputs an ordinary intervention experiment exposes.
        visible = {
            "target_token": result["target_token"],
            "target_probability": result["restored_target_probability"],
            "top_token": result["restored_top_token"],
            "target_is_top": result["restored_target_is_top"],
        }
        return {**visible, "experiments_remaining": self.remaining}

if __name__ == "__main__":
    RestorationTools.run()
