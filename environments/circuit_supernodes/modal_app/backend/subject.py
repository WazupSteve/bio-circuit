"""Transport-agnostic subject backend: one loaded model, five operations."""


class SubjectBackend:
    """Loads Qwen + transcoders once and serves every subject operation."""

    def __init__(
        self,
        model_name: str,
        transcoder_set: str,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.98,
    ) -> None:
        import torch
        from circuit_tracer import ReplacementModel

        from .attribution import AttributionEngine
        from .corruption import CorruptionEngine
        from .interventions import InterventionEvaluator

        self.model = ReplacementModel.from_pretrained(
            model_name, transcoder_set, backend="nnsight", dtype=torch.bfloat16
        )
        self.intervention_evaluator = InterventionEvaluator(self.model)
        self.attribution = AttributionEngine(self.model, node_threshold, edge_threshold)
        self.corruption = CorruptionEngine(self.model, self.intervention_evaluator)

    def _logits(self, tokens):
        batch = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        return self.model._model(input_ids=batch).logits

    def sample_until_answer(self, request: dict) -> dict:
        from circuit_supernodes.runtime.sampling import sample_until_answer

        return sample_until_answer(self.model, request, logits_fn=self._logits)

    def restore_output(self, prompt, corruption, restorations, target_token):
        return self.intervention_evaluator.restore_output(
            prompt, corruption, restorations, target_token
        )

    def evaluate_ablation_cohort(self, positives, controls):
        return self.intervention_evaluator.evaluate_ablation_cohort(positives, controls)

    def test_corruption_coordinates(
        self, prompt, expected_target_token, coordinate_spec
    ):
        return self.corruption.corrupt_with_coordinate_spec(
            prompt, expected_target_token, coordinate_spec
        )

    def attribute_prompt(self, prompt):
        try:
            return self.attribution.attribute_graph(prompt)
        except Exception as exc:
            print(f"attribution failed for {prompt!r}: {exc}")
            return None


METHODS = (
    "sample_until_answer",
    "restore_output",
    "evaluate_ablation_cohort",
    "test_corruption_coordinates",
    "attribute_prompt",
)
