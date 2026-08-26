"""Thin Modal RPC surface over the shared SubjectBackend."""

import modal

from .deployment import app, hf_cache, image
from .subject import SubjectBackend


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
        self.backend = SubjectBackend(
            self.model_name,
            self.transcoder_set,
            float(self.node_threshold),
            float(self.edge_threshold),
        )

    @modal.method()
    def sample_until_answer(self, request: dict) -> dict:
        return self.backend.sample_until_answer(request)

    @modal.method()
    def restore_output(self, prompt, corruption, restorations, target_token):
        return self.backend.restore_output(
            prompt, corruption, restorations, target_token
        )

    @modal.method()
    def evaluate_ablation_cohort(self, positives, controls):
        return self.backend.evaluate_ablation_cohort(positives, controls)

    @modal.method()
    def test_corruption_coordinates(
        self, prompt, expected_target_token, coordinate_spec
    ):
        return self.backend.test_corruption_coordinates(
            prompt, expected_target_token, coordinate_spec
        )

    @modal.method()
    def attribute_prompt(self, prompt):
        return self.backend.attribute_prompt(prompt)
