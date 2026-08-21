"""Typed, task-agnostic adapters for model-facing Circuit Tracer tools."""

from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field, FiniteFloat

from ..models import StrictModel, TracerGraphPayload
from .graphs import resolve_interventions, stable_feature_id, top_outputs
from .modal import ModalSubject, retry_modal_call
from .responses import render_feature_view


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
        self._subject: ModalSubject | None = None
        self._corruption_spec: list[tuple[int, int, float]] = []

    async def setup_task(self, task) -> None:
        self._subject = ModalSubject(task.modal_app, task.modal_cls)
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
        clean = TracerGraphPayload.model_validate(raw)
        if not self._corruption_spec:
            return clean
        outputs = top_outputs(clean)
        if not outputs:
            raise ValueError("prompt has no attributable top token")
        raw = await self._call(
            "build_corrupted_sample_from_spec",
            prompt,
            outputs[0][0],
            self._corruption_spec,
        )
        if not isinstance(raw, dict) or "graph" not in raw:
            raise ValueError("Circuit Tracer could not apply the task corruption")
        return TracerGraphPayload.model_validate(raw["graph"])

    @vf.tool(name="attribute")
    async def attribute(self, prompt: str, feature_count: int) -> dict:
        """Attribute any prompt and return an importance-ranked graph view."""
        try:
            graph = await self._graph(prompt)
            return render_feature_view(graph, feature_count)
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
            return raw["positives"][0]
        except Exception as exc:
            return {"error": validation_message(exc)}
