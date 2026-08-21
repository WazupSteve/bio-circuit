"""GPU-backend mechanics shared by the Modal subject services."""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from typing import Any

from .ablation_contract import ablation_result, evaluate_cohorts, invalid_coordinates
from .model_ops import normalize_activations, tokens_and_target


def export_attribution_graph(
    model: Any,
    prompt: str,
    *,
    max_feature_nodes: int,
    node_threshold: float,
    edge_threshold: float,
) -> dict[str, Any]:
    """Attribute a prompt and serialize its graph without shared temp-file state."""

    import torch  # ty: ignore[unresolved-import] -- installed in the Modal image
    from circuit_tracer import attribute  # ty: ignore[unresolved-import]
    from circuit_tracer.utils.create_graph_files import (  # ty: ignore[unresolved-import]
        create_graph_files,
    )

    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=5,
        desired_logit_prob=0.95,
        max_feature_nodes=max_feature_nodes,
        batch_size=64,
        offload=None,
        verbose=True,
    )
    try:
        with TemporaryDirectory(prefix="circuit-graph-") as directory:
            slug = "graph"
            create_graph_files(
                graph_or_path=graph,
                slug=slug,
                output_path=directory,
                node_threshold=node_threshold,
                edge_threshold=edge_threshold,
            )
            with open(f"{directory}/{slug}.json", encoding="utf-8") as stream:
                return json.load(stream)
    finally:
        del graph
        torch.cuda.empty_cache()


def evaluate_clean_ablation_cohort(
    model: Any,
    positives: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    *,
    require_single_token: bool,
    require_round_trip: bool,
    validate_activation_shape: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Apply multiplicative interventions against an uncorrupted baseline."""

    import torch  # ty: ignore[unresolved-import] -- installed in the Modal image

    def evaluate(example: dict[str, Any], *, positive: bool) -> dict[str, Any]:
        prompt = str(example["prompt"])
        expected_token = str(example["expected_token"])
        interventions = [
            (int(layer), int(feature), int(position), float(factor))
            for layer, feature, position, factor in example.get("interventions", [])
        ]
        tokens, target_id = tokens_and_target(
            model,
            prompt,
            expected_token,
            require_single=require_single_token,
            require_round_trip=require_round_trip,
        )
        with torch.inference_mode():
            clean_logits, activations = model.get_activations(tokens)
            activations = normalize_activations(
                activations, validate_shape=validate_activation_shape
            )
            layers, positions, features = activations.shape
            invalid = invalid_coordinates(
                [item[:3] for item in interventions],
                layers=layers,
                positions=positions,
                features=features,
            )
            if invalid:
                raise ValueError(
                    f"invalid cohort intervention coordinates: {invalid[:4]}"
                )
            absolute = [
                (
                    layer,
                    position,
                    feature,
                    activations[layer, position, feature].float().item() * factor,
                )
                for layer, feature, position, factor in interventions
            ]
            changed_logits = clean_logits
            if absolute:
                changed_logits, _ = model.feature_intervention(
                    tokens, absolute, return_activations=False
                )
            clean_final = clean_logits.reshape(-1, clean_logits.shape[-1])[-1].float()
            changed_final = changed_logits.reshape(-1, changed_logits.shape[-1])[
                -1
            ].float()
            changed_probabilities = torch.softmax(changed_final, dim=-1)
            return ablation_result(
                expected_token=expected_token,
                expected_id=target_id,
                baseline_top=int(clean_final.argmax().item()),
                intervened_top=int(changed_final.argmax().item()),
                decode=lambda token_id: model.tokenizer.decode([token_id]),
                positive=positive,
                intervened_expected_probability=float(
                    changed_probabilities[target_id].item()
                ),
            )

    return evaluate_cohorts(positives, controls, evaluate)
