"""Factory for the ordinary attribution Modal services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import modal
from circuit_supernodes.runtime.graphs import MAX_FEATURE_NODES
from circuit_supernodes.runtime.modal_backend import (
    evaluate_clean_ablation_cohort,
    export_attribution_graph,
)


def create_attributor_service(
    *,
    app: modal.App,
    class_name: str,
    image: modal.Image,
    model_name: str,
    transcoder_set: str,
    node_threshold: float,
    edge_threshold: float,
    volume: modal.Volume,
    max_containers: int,
    scaledown_window: int,
    secrets: list[modal.Secret] | None = None,
    model_options: dict[str, Any] | None = None,
    logits_adapter: Callable[[Any, Any], Any] | None = None,
    require_single_token: bool = False,
    require_round_trip: bool = False,
    validate_activation_shape: bool = False,
    forgive_attribution_errors: bool = False,
):
    """Create one model-configured sampling, attribution, and ablation service."""

    class AttributorService:
        @modal.enter()
        def load(self) -> None:
            import torch
            from circuit_tracer import ReplacementModel

            self.model = ReplacementModel.from_pretrained(
                model_name,
                transcoder_set,
                dtype=torch.bfloat16,
                **(model_options or {}),
            )

        @modal.method()
        def sample_until_answer(self, request: dict) -> dict:
            from circuit_supernodes.runtime.sampling import sample_until_answer

            logits_fn = None
            if logits_adapter is not None:

                def logits_fn(tokens):
                    return logits_adapter(self.model, tokens)

            return sample_until_answer(self.model, request, logits_fn=logits_fn)

        @modal.method()
        def attribute_prompt(self, prompt: str) -> dict | None:
            try:
                return export_attribution_graph(
                    self.model,
                    prompt,
                    max_feature_nodes=MAX_FEATURE_NODES,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )
            except Exception as exc:
                if not forgive_attribution_errors:
                    raise
                print(f"attribution failed for {prompt!r}: {exc}")
                return None

        @modal.method()
        def evaluate_ablation_cohort(
            self,
            positives: list[dict],
            controls: list[dict],
        ) -> dict:
            return evaluate_clean_ablation_cohort(
                self.model,
                positives,
                controls,
                require_single_token=require_single_token,
                require_round_trip=require_round_trip,
                validate_activation_shape=validate_activation_shape,
            )

    AttributorService.__name__ = class_name
    AttributorService.__qualname__ = class_name
    return app.cls(
        image=image,
        gpu="A100-80GB",
        timeout=60 * 60,
        volumes={"/root/.cache/huggingface": volume},
        secrets=secrets or [],
        max_containers=max_containers,
        scaledown_window=scaledown_window,
    )(AttributorService)
