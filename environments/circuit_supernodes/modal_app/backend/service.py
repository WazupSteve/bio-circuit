"""Thin Modal RPC surface for the unified backend."""

import modal

from .attribution import AttributionEngine
from .corruption import CorruptionEngine
from .deployment import app, hf_cache, image
from .interventions import InterventionEvaluator


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    max_containers=8,
    scaledown_window=600,
)
class Attributor:
    model_name: str = modal.parameter()
    transcoder_set: str = modal.parameter()
    # Modal class parameters do not support floats, so thresholds cross the
    # class boundary as strings and are converted when the container starts.
    node_threshold: str = modal.parameter(default="0.8")
    edge_threshold: str = modal.parameter(default="0.98")

    @modal.enter()
    def load(self):
        import torch
        from circuit_tracer import ReplacementModel

        node_threshold = float(self.node_threshold)
        edge_threshold = float(self.edge_threshold)

        self.model = ReplacementModel.from_pretrained(
            self.model_name,
            self.transcoder_set,
            backend="nnsight",
            dtype=torch.bfloat16,
        )
        self.intervention_evaluator = InterventionEvaluator(self.model)
        self.attribution = AttributionEngine(self.model, node_threshold, edge_threshold)
        self.corruption = CorruptionEngine(self.model, self.intervention_evaluator)

    def _logits(self, tokens):
        batch = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        return self.model._model(input_ids=batch).logits

    @modal.method()
    def sample_until_answer(self, request: dict) -> dict:
        from circuit_supernodes.runtime.sampling import sample_until_answer

        return sample_until_answer(self.model, request, logits_fn=self._logits)

    @modal.method()
    def restore_output(self, prompt, corruption, restorations, target_token):
        return self.intervention_evaluator.restore_output(
            prompt, corruption, restorations, target_token
        )

    @modal.method()
    def evaluate_ablation_cohort(self, positives, controls):
        return self.intervention_evaluator.evaluate_ablation_cohort(positives, controls)

    @modal.method()
    def test_corruption_coordinates(
        self, prompt, expected_target_token, coordinate_spec
    ):
        return self.corruption.corrupt_with_coordinate_spec(
            prompt, expected_target_token, coordinate_spec
        )

    @modal.method()
    def attribute_prompt(self, prompt):
        try:
            return self.attribution.attribute_graph(prompt)
        except Exception as exc:
            print(f"attribution failed for {prompt!r}: {exc}")
            return None
