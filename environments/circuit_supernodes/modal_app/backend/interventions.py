"""Model replay, multiplicative interventions, and cohort evaluation."""

from circuit_supernodes.runtime.ablation_contract import (
    ablation_result,
    evaluate_cohorts,
)
from circuit_supernodes.runtime.corruption import (
    MAX_CORRUPTION_FACTOR,
    MAX_CORRUPTION_NODES,
    MIN_CORRUPTION_FACTOR,
    MIN_CORRUPTION_NODES,
    InterventionPayload,
    compose_multipliers,
)
from circuit_supernodes.runtime.model_ops import (
    normalize_activations,
    target_probability,
    tokens_and_target,
)

MAX_INTERVENTIONS = 60


class InterventionEvaluator:
    def __init__(self, model):
        self.model = model

    def tokens_and_target(self, prompt: str, target_token: str):
        return tokens_and_target(
            self.model,
            prompt,
            target_token,
            require_single=True,
            require_round_trip=True,
        )

    def intervention_input(self, tokens) -> str:
        """Render the BOS-prefixed tensor back to text for NNSight interventions."""

        rendered = self.model.tokenizer.decode(
            tokens.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        roundtrip = self.model.tokenizer.encode(rendered, add_special_tokens=False)
        if roundtrip != tokens.tolist():
            raise ValueError("BOS-prefixed intervention input does not round-trip")
        return rendered

    @staticmethod
    def normalize_acts(acts):
        return normalize_activations(acts)

    def restore_output(
        self,
        prompt: str,
        corruption: list[tuple[int, int, int, float]],
        restorations: list[tuple[int, int, int, float]],
        target_token: str,
    ) -> dict:
        """Replay hidden corruption and compose submitted multipliers on top.

        For clean activation ``a``, hidden factor ``c`` and submitted factor
        ``r``, the executed value is ``a * c * r``. Missing factors are one, so
        submitting weight one cannot reveal or undo the hidden intervention.
        """

        import torch

        corruption = list(
            InterventionPayload.model_validate(
                corruption,
                context={
                    "label": "corruption",
                    "min_count": MIN_CORRUPTION_NODES,
                    "max_count": MAX_CORRUPTION_NODES,
                    "min_factor": MIN_CORRUPTION_FACTOR,
                    "max_abs_factor": MAX_CORRUPTION_FACTOR,
                },
            ).root
        )
        restorations = list(
            InterventionPayload.model_validate(
                restorations,
                context={
                    "label": "restoration",
                    "max_count": MAX_INTERVENTIONS,
                },
            ).root
        )

        tokens, target_id = self.tokens_and_target(prompt, target_token)
        intervention_input = self.intervention_input(tokens)
        with torch.no_grad():
            standard_logits, base_acts = self.model.get_activations(tokens)
            base_acts = self.normalize_acts(base_acts)
            hidden_factors = {item[:3]: item[3] for item in corruption}
            composed_factors = compose_multipliers(corruption, restorations)

            def interventions(include_submission: bool, activations=base_acts):
                factors = composed_factors if include_submission else hidden_factors
                values = []
                for (layer, feature, position), factor in sorted(factors.items()):
                    base = activations[layer, position, feature].float().item()
                    values.append((layer, position, feature, base * factor))
                return values

            corrupted_logits, _ = self.model.feature_intervention(
                intervention_input,
                interventions(False),
                freeze_attention=False,
                return_activations=False,
            )
            restored_logits, _ = self.model.feature_intervention(
                intervention_input,
                interventions(True),
                freeze_attention=False,
                return_activations=False,
            )

            def final_logits(logits):
                return logits.reshape(-1, logits.shape[-1])[-1].float()

            standard_final = final_logits(standard_logits)
            corrupted_final = final_logits(corrupted_logits)
            restored_final = final_logits(restored_logits)
            standard_log_probs = torch.log_softmax(standard_final, dim=-1)

            def standard_kl(candidate) -> float:
                import torch.nn.functional as functional

                candidate_log_probs = torch.log_softmax(candidate, dim=-1)
                return functional.kl_div(
                    candidate_log_probs,
                    standard_log_probs,
                    reduction="sum",
                    log_target=True,
                ).item()

            standard_top_id = int(standard_final.argmax().item())
            corrupted_top_id = int(corrupted_final.argmax().item())
            restored_top_id = int(restored_final.argmax().item())
            out = {
                "target_token": target_token,
                "standard_target_probability": target_probability(
                    standard_logits, target_id
                ),
                "corrupted_target_probability": target_probability(
                    corrupted_logits, target_id
                ),
                "restored_target_probability": target_probability(
                    restored_logits, target_id
                ),
                "corrupted_standard_kl": standard_kl(corrupted_final),
                "restored_standard_kl": standard_kl(restored_final),
                "standard_top_token": self.model.tokenizer.decode([standard_top_id]),
                "corrupted_top_token": self.model.tokenizer.decode([corrupted_top_id]),
                "restored_top_token": self.model.tokenizer.decode([restored_top_id]),
                "standard_target_is_top": standard_top_id == target_id,
                "corruption_changed_top_token": corrupted_top_id != standard_top_id,
                "restored_target_is_top": restored_top_id == target_id,
            }
            del standard_logits, corrupted_logits, restored_logits, base_acts
        return out

    def evaluate_ablation_cohort(
        self, positives: list[dict], controls: list[dict]
    ) -> dict:
        """Test a negated proposal relative to each corrupted baseline."""

        import torch

        def evaluate(example: dict, *, positive: bool) -> dict:
            prompt = str(example["prompt"])
            expected_token = str(example["expected_token"])
            baseline_raw = example.get("baseline_interventions", [])
            baseline = (
                list(
                    InterventionPayload.model_validate(
                        baseline_raw,
                        context={
                            "label": "baseline corruption",
                            "min_count": MIN_CORRUPTION_NODES,
                            "max_count": MAX_CORRUPTION_NODES,
                            "min_factor": MIN_CORRUPTION_FACTOR,
                            "max_abs_factor": MAX_CORRUPTION_FACTOR,
                        },
                    ).root
                )
                if baseline_raw
                else []
            )
            interventions = list(
                InterventionPayload.model_validate(
                    example.get("interventions", []),
                    context={
                        "label": "ablation intervention",
                        "max_count": MAX_INTERVENTIONS,
                    },
                ).root
            )
            tokens, expected_id = self.tokens_and_target(prompt, expected_token)
            intervention_input = self.intervention_input(tokens)
            with torch.no_grad():
                clean_logits, base_acts = self.model.get_activations(tokens)
                base_acts = self.normalize_acts(base_acts)
                baseline_factors = {item[:3]: item[3] for item in baseline}
                changed_factors = compose_multipliers(baseline, interventions)

                def run(factors, activations=base_acts, baseline_logits=clean_logits):
                    if not factors:
                        clean_final = baseline_logits.reshape(
                            -1, baseline_logits.shape[-1]
                        )[-1].float()
                        return int(clean_final.argmax().item())
                    absolute = [
                        (
                            layer,
                            position,
                            feature,
                            activations[layer, position, feature].float().item()
                            * factor,
                        )
                        for (layer, feature, position), factor in sorted(
                            factors.items()
                        )
                    ]
                    logits, _ = self.model.feature_intervention(
                        intervention_input,
                        absolute,
                        freeze_attention=False,
                        return_activations=False,
                    )
                    final = logits.reshape(-1, logits.shape[-1])[-1].float()
                    return int(final.argmax().item())

                baseline_top = run(baseline_factors)
                changed_top = run(changed_factors)
                result = ablation_result(
                    expected_token=expected_token,
                    expected_id=expected_id,
                    baseline_top=baseline_top,
                    intervened_top=changed_top,
                    decode=lambda token_id: self.model.tokenizer.decode([token_id]),
                    positive=positive,
                )
                del clean_logits, base_acts
                return result

        return evaluate_cohorts(positives, controls, evaluate)
