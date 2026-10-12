"""Typed, task-agnostic adapters for model-facing Circuit Tracer tools."""

import asyncio

import verifiers.v1 as vf
from pydantic import Field, FiniteFloat

from ..models import StrictModel, TracerGraphPayload
from .graph_store import GraphStore, QueryView
from .graphs import resolve_interventions, stable_feature_id, top_outputs
from .modal import Subject, retry_modal_call, subject_from_config


def validation_message(exc: Exception) -> str:
    """Return the first concise Pydantic error message."""

    errors = getattr(exc, "errors", None)
    if callable(errors):
        rows = errors(include_url=False, include_input=False)
        if rows:
            return str(rows[0]["msg"]).removeprefix("Value error, ")
    return str(exc)


class InterventionToolRequest(StrictModel):
    source_prompt: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    intervention_weight: FiniteFloat


class CircuitExperimentTools(vf.Toolset[vf.ToolsetConfig, vf.State]):
    """Shared arbitrary-prompt Circuit Tracer interface for every taskset."""

    TOOL_PREFIX = "circuit"

    def __init__(self, config: vf.ToolsetConfig) -> None:
        super().__init__(config)
        self._subject: Subject | None = None
        self._corruption_spec: list[tuple[int, int, float]] = []
        self._graph_store: GraphStore | None = None
        self._graph_lock = asyncio.Lock()

    async def setup_task(self, task) -> None:
        async with self._graph_lock:
            self._clear_graph()
        self._subject = subject_from_config(task.subject_config)
        self._corruption_spec = list(task.corruption_spec)

    @retry_modal_call
    async def _call(self, method: str, *args):
        if self._subject is None:
            raise RuntimeError("toolset has not been initialized")
        return await self._subject.call(method, *args)

    async def _graph(self, prompt: str) -> TracerGraphPayload:
        raw = await self._call("attribute_prompt", prompt)
        if raw is None:
            raise ValueError("Circuit Tracer could not attribute the prompt")
        return TracerGraphPayload.model_validate(raw)

    @vf.tool(name="attribute")
    async def attribute(self, prompt: str) -> dict:
        """Store a prompt's graph and return its ID and summary for query_graph.

        Deletes the previous stored graph before attribution, even if this request
        fails. Use query_graph for ranked nodes, connections, and prompt tokens.
        """
        async with self._graph_lock:
            self._clear_graph()
            try:
                graph = await self._graph(prompt)
                self._graph_store = GraphStore()
                return self._graph_store.write(
                    graph, include_outputs=not self._corruption_spec
                )
            except Exception as exc:
                self._clear_graph()
                return {"error": validation_message(exc)}

    def _clear_graph(self) -> None:
        if self._graph_store is not None:
            self._graph_store.close()
            self._graph_store = None

    @vf.tool(name="query_graph")
    async def query_graph(
        self,
        graph_id: str,
        view: QueryView = "nodes",
        offset: int = 0,
        limit: int = 20,
        node_id: str | None = None,
        kind: str | None = None,
    ) -> dict:
        """Read a bounded page from the latest attributed graph.

        Nodes are ranked by influence delta; filter by kind (feature, embedding,
        error, logit), exact node_id, or stable layer_feature ID. Edges are ranked
        by absolute weight; node_id filters both incoming and outgoing edges and
        must be an exact positioned ID. Tokens are ordered by position. Follow
        next_offset with the same filters. Limit is 1–50; responses also have a
        byte budget and text fields are truncated to 300 characters. A new
        attribute call invalidates the previous graph_id. Intervene does not.
        """
        try:
            async with self._graph_lock:
                if self._graph_store is None:
                    raise ValueError("No stored graph; call attribute first")
                if graph_id != self._graph_store.graph_id:
                    raise ValueError("Graph has been replaced; use the latest graph_id")
                return self._graph_store.query(
                    view=view,
                    offset=offset,
                    limit=limit,
                    node_id=node_id,
                    kind=kind,
                )
        except Exception as exc:
            return {"error": validation_message(exc)}

    @vf.tool(name="intervene")
    async def intervene(
        self,
        source_prompt: str,
        node_id: str,
        prompt: str,
        intervention_weight: float,
    ) -> dict:
        """Resolve a source node on another prompt and report changed top tokens."""
        try:
            request = InterventionToolRequest.model_validate(
                {
                    "source_prompt": source_prompt,
                    "node_id": node_id,
                    "prompt": prompt,
                    "intervention_weight": intervention_weight,
                }
            )
            source = await self._graph(request.source_prompt)
            stable_id = stable_feature_id(request.node_id)
            if not resolve_interventions(source, [(stable_id, 1.0)]):
                raise ValueError(f"FEATURE {stable_id!r} is absent from source_prompt")
            del source
            evaluation = await self._graph(request.prompt)
            interventions = resolve_interventions(
                evaluation, [(stable_id, request.intervention_weight)]
            )
            if not interventions:
                raise ValueError(f"FEATURE {stable_id!r} is absent from prompt")
            outputs = top_outputs(evaluation)
            if not outputs:
                raise ValueError("prompt has no attributable top token")
            baseline = resolve_interventions(
                evaluation,
                [
                    (f"{layer}_{feature}", factor)
                    for layer, feature, factor in self._corruption_spec
                ],
            )
            raw = await self._call(
                "evaluate_ablation_cohort",
                [
                    {
                        "prompt": request.prompt,
                        "expected_token": outputs[0][0],
                        "interventions": interventions,
                        "baseline_interventions": baseline,
                    }
                ],
                [],
            )
            result = raw["positives"][0]
            if not self._corruption_spec:
                return result
            baseline = result["baseline_top_token"]
            intervened = result["intervened_top_token"]
            return {
                "baseline_top_token": baseline,
                "intervened_top_token": intervened,
                "changed_top_token": baseline != intervened,
            }
        except Exception as exc:
            return {"error": validation_message(exc)}


if __name__ == "__main__":
    CircuitExperimentTools.run()
