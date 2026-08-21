"""Corruption screening for stored clean-graph coordinates."""

from circuit_supernodes.runtime.corruption import (
    MAX_CORRUPTION_FACTOR,
    MAX_CORRUPTION_NODES,
    MIN_CORRUPTION_NODES,
    InterventionPayload,
)


class CorruptionEngine:
    def __init__(self, model, intervention_evaluator):
        self.model = model
        self.intervention_evaluator = intervention_evaluator

    def corrupt_with_coordinate_spec(
        self,
        prompt: str,
        expected_target_token: str,
        coordinate_spec: list[tuple[int, int, int, float]],
    ) -> dict:
        """Apply corruption coordinates already resolved from a stored clean graph."""

        import torch

        selected = list(
            InterventionPayload.model_validate(
                [
                    (int(layer), int(feature), int(position), float(factor))
                    for layer, feature, position, factor in coordinate_spec
                ][:MAX_CORRUPTION_NODES],
                context={
                    "label": "corruption",
                    "min_count": MIN_CORRUPTION_NODES,
                    "max_count": MAX_CORRUPTION_NODES,
                    "max_abs_factor": MAX_CORRUPTION_FACTOR,
                },
            ).root
        )
        if len(selected) < MIN_CORRUPTION_NODES:
            return {"rejected": "too few corruption coordinates in prompt"}

        tokens = self.model.ensure_tokenized(prompt)
        intervention_input = self.intervention_evaluator.intervention_input(tokens)
        with torch.no_grad():
            clean_logits, clean_acts = self.model.get_activations(tokens)
            clean_acts = self.intervention_evaluator.normalize_acts(clean_acts)
            clean_final = clean_logits.reshape(-1, clean_logits.shape[-1])[-1]
            target_id = int(clean_final.argmax().item())
            target_token = self.model.tokenizer.decode([target_id])
            if not target_token or target_token != expected_target_token:
                return {"rejected": "clean target mismatch"}
            absolute = [
                (
                    layer,
                    position,
                    feature,
                    clean_acts[layer, position, feature].float().item() * factor,
                )
                for layer, feature, position, factor in selected
            ]
            corrupted_logits, _ = self.model.feature_intervention(
                intervention_input,
                absolute,
                freeze_attention=False,
                return_activations=False,
            )
            corrupted_final = corrupted_logits.reshape(-1, corrupted_logits.shape[-1])[
                -1
            ]
            changed = int(corrupted_final.argmax().item()) != target_id

        result = {
            "changed_top_token": changed,
            "target_token": target_token,
            "corruption": selected,
        }
        return result
