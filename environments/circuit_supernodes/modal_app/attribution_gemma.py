"""Gemma-2-2B circuit tracing and functional-replacement grading."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import modal

MODEL_NAME = "unsloth/gemma-2-2b"
TRANSCODER_SET = "mntss/gemma-scope-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_FEATURE_NODES = 4096

app = modal.App("circuit-supernodes-gemma-attribution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/safety-research/circuit-tracer.git@v0.5.2",
        "cloudpickle==3.1.1",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("circuit_supernodes")
)

hf_cache = modal.Volume.from_name("circuit-supernodes-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=1,
    scaledown_window=300,
)
class GemmaAttributor:
    @modal.enter()
    def load(self) -> None:
        import torch
        from circuit_tracer import ReplacementModel

        self.model = ReplacementModel.from_pretrained(
            MODEL_NAME,
            TRANSCODER_SET,
            backend="nnsight",
            dtype=torch.bfloat16,
        )

    def _tokens_and_target(self, prompt: str, target_token: str):
        token_ids = self.model.tokenizer.encode(target_token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"target token {target_token!r} must encode to exactly one token; "
                f"got {len(token_ids)}"
            )
        return self.model.ensure_tokenized(prompt), token_ids[0]

    @staticmethod
    def _normalize_acts(activations):
        normalized = activations[:, 0] if activations.dim() == 4 else activations
        if normalized.dim() != 3:
            raise ValueError(
                "expected activation shape [layers, positions, features], "
                f"got {tuple(normalized.shape)}"
            )
        return normalized

    @staticmethod
    def _canonical_coordinates(
        name: str, coordinates: list[tuple[int, int, int]]
    ) -> list[tuple[int, int, int]]:
        malformed = [
            coordinate
            for coordinate in coordinates
            if len(coordinate) != 3
            or any(type(value) is not int for value in coordinate)
        ]
        if malformed:
            raise ValueError(f"malformed {name} coordinates: {malformed[:4]}")
        return sorted(set(map(tuple, coordinates)))

    @modal.method()
    def sample_until_answer(self, request: dict) -> dict:
        """Use the same bounded answer-aware sampler as every task family."""

        from circuit_supernodes.sampling import sample_until_answer

        return sample_until_answer(self.model, request)

    @modal.method()
    def attribute_prompt(self, prompt: str) -> dict:
        """Run the subject model and export one fresh attribution graph."""

        import torch
        from circuit_tracer import attribute
        from circuit_tracer.utils.create_graph_files import create_graph_files

        graph = attribute(
            prompt=prompt,
            model=self.model,
            max_n_logits=5,
            desired_logit_prob=0.95,
            max_feature_nodes=MAX_FEATURE_NODES,
            batch_size=64,
            offload=None,
            verbose=True,
        )
        try:
            with TemporaryDirectory(prefix="gemma-graph-") as directory:
                output = Path(directory)
                slug = "graph"
                create_graph_files(
                    graph_or_path=graph,
                    slug=slug,
                    output_path=str(output),
                    node_threshold=NODE_THRESHOLD,
                    edge_threshold=EDGE_THRESHOLD,
                )
                return json.loads((output / f"{slug}.json").read_text())
        finally:
            del graph
            torch.cuda.empty_cache()

    @modal.method()
    def compare_feature_ablations(
        self,
        prompt: str,
        reference_features: list[tuple[int, int, int]],
        replacement_features: list[tuple[int, int, int]],
        target_token: str,
    ) -> dict:
        """Compare next-token distributions after ablating two feature sets."""

        import torch

        reference_features = self._canonical_coordinates(
            "reference", reference_features
        )
        replacement_features = self._canonical_coordinates(
            "replacement", replacement_features
        )
        if not reference_features or not replacement_features:
            return {"valid": False, "error": "feature sets must be non-empty"}

        tokens, target_id = self._tokens_and_target(prompt, target_token)
        with torch.no_grad():
            base_logits, base_acts = self.model.get_activations(tokens)
            base_acts = self._normalize_acts(base_acts)
            layers, positions, features = base_acts.shape

            invalid_reference = self._out_of_bounds(
                reference_features, layers, positions, features
            )
            if invalid_reference:
                raise ValueError(
                    f"invalid hidden reference coordinates: {invalid_reference[:4]}"
                )
            invalid_replacement = self._out_of_bounds(
                replacement_features, layers, positions, features
            )
            if invalid_replacement:
                return {
                    "valid": False,
                    "error": (
                        f"invalid replacement coordinates: {invalid_replacement[:4]}"
                    ),
                }

            def ablate(coordinates: list[tuple[int, int, int]]):
                interventions = [
                    (layer, position, feature, 0.0)
                    for layer, feature, position in coordinates
                ]
                logits, _ = self.model.feature_intervention(
                    tokens, interventions, return_activations=False
                )
                return logits.reshape(-1, logits.shape[-1])[-1].float()

            base_final = base_logits.reshape(-1, base_logits.shape[-1])[-1].float()
            reference_final = ablate(reference_features)
            replacement_final = ablate(replacement_features)
            base_probs = torch.softmax(base_final, dim=-1)
            reference_probs = torch.softmax(reference_final, dim=-1)
            replacement_probs = torch.softmax(replacement_final, dim=-1)

            output_tv = 0.5 * torch.abs(reference_probs - replacement_probs).sum()
            reference_effect_tv = 0.5 * torch.abs(base_probs - reference_probs).sum()
            replacement_effect_tv = (
                0.5 * torch.abs(base_probs - replacement_probs).sum()
            )
            reference_top_id = int(reference_probs.argmax().item())
            replacement_top_id = int(replacement_probs.argmax().item())

            return {
                "valid": True,
                "top_token_match": reference_top_id == replacement_top_id,
                "reference_top_token": self.model.tokenizer.decode([reference_top_id]),
                "replacement_top_token": self.model.tokenizer.decode(
                    [replacement_top_id]
                ),
                "output_total_variation": float(output_tv.item()),
                "reference_effect_total_variation": float(reference_effect_tv.item()),
                "replacement_effect_total_variation": float(
                    replacement_effect_tv.item()
                ),
                "target_token": target_token,
                "reference_target_probability": float(
                    reference_probs[target_id].item()
                ),
                "replacement_target_probability": float(
                    replacement_probs[target_id].item()
                ),
                "target_probability_gap": float(
                    torch.abs(
                        reference_probs[target_id] - replacement_probs[target_id]
                    ).item()
                ),
            }

    @modal.method()
    def evaluate_intervention_cohort(
        self, cohort: list[dict], controls: list[dict]
    ) -> dict:
        """Grade causal restoration on goal prompts and specificity controls."""

        import torch

        def evaluate(example: dict, *, control: bool) -> dict:
            prompt = str(example["prompt"])
            target_token = str(example["target_token"])
            raw = example.get("interventions", [])
            interventions = [
                (int(layer), int(feature), int(position), float(weight))
                for layer, feature, position, weight in raw
            ]
            tokens, target_id = self._tokens_and_target(prompt, target_token)
            with torch.no_grad():
                clean_logits, activations = self.model.get_activations(tokens)
                activations = self._normalize_acts(activations)
                layers, positions, features = activations.shape
                coordinates = [item[:3] for item in interventions]
                invalid = self._out_of_bounds(coordinates, layers, positions, features)
                if invalid:
                    raise ValueError(
                        f"invalid cohort intervention coordinates: {invalid[:4]}"
                    )

                def run(
                    weights: list[tuple[int, int, int, float]],
                    base_activations=activations,
                ):
                    absolute = [
                        (
                            layer,
                            position,
                            feature,
                            base_activations[layer, position, feature].float().item()
                            * weight,
                        )
                        for layer, feature, position, weight in weights
                    ]
                    logits, _ = self.model.feature_intervention(
                        tokens, absolute, return_activations=False
                    )
                    final = logits.reshape(-1, logits.shape[-1])[-1].float()
                    probabilities = torch.softmax(final, dim=-1)
                    return int(probabilities.argmax().item()), float(
                        probabilities[target_id].item()
                    )

                intervened_top, intervened_probability = run(interventions)
                result = {
                    "target_token": target_token,
                    "intervened_target_is_top": intervened_top == target_id,
                    "intervened_target_probability": intervened_probability,
                    "intervened_top_token": self.model.tokenizer.decode(
                        [intervened_top]
                    ),
                }
                if not control:
                    ablated = [(*item[:3], 0.0) for item in interventions]
                    baseline_top, baseline_probability = run(ablated)
                    result.update(
                        {
                            "baseline_target_is_top": baseline_top == target_id,
                            "baseline_target_probability": baseline_probability,
                            "shifted_to_target": baseline_top != target_id
                            and intervened_top == target_id,
                        }
                    )
                del clean_logits, activations
                return result

        return {
            "cohort": [evaluate(example, control=False) for example in cohort],
            "controls": [evaluate(example, control=True) for example in controls],
        }

    @staticmethod
    def _out_of_bounds(
        coordinates: list[tuple[int, int, int]],
        layers: int,
        positions: int,
        features: int,
    ) -> list[tuple[int, int, int]]:
        return [
            coordinate
            for coordinate in coordinates
            if not 0 <= coordinate[0] < layers
            or not 0 <= coordinate[1] < features
            or not 0 <= coordinate[2] < positions
        ]
