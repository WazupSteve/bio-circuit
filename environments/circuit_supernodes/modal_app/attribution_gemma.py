"""Gemma-2-2B backend for the fixed causal-response-bank audit."""

import hashlib
import json
import math

import modal

MODEL_NAME = "unsloth/gemma-2-2b"
TRANSCODER_SET = "mntss/gemma-scope-transcoders"
RESPONSE_BANK_VERSION = "v1-local-derivative"
CALIBRATION_FRACTIONS = (0.25, 0.5)
VALIDATION_FRACTIONS = (0.75, 1.0)

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
    def load(self):
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
        if not token_ids:
            raise ValueError(f"target token {target_token!r} encodes to nothing")
        return self.model.ensure_tokenized(prompt), token_ids[0]

    @staticmethod
    def _target_log_odds(logits, target_id: int) -> float:
        import torch

        final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
        other_logits = torch.cat(
            (final_logits[:target_id], final_logits[target_id + 1 :])
        )
        return (final_logits[target_id] - torch.logsumexp(other_logits, dim=0)).item()

    @staticmethod
    def _normalize_acts(acts):
        return acts[:, 0] if acts.dim() == 4 else acts

    @modal.method()
    def target_probability(self, prompt: str, target_token: str) -> float:
        """Return the unperturbed next-token probability for source validation."""

        import torch

        tokens, target_id = self._tokens_and_target(prompt, target_token)
        with torch.no_grad():
            logits, _ = self.model.get_activations(tokens)
            final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
            return torch.softmax(final_logits, dim=-1)[target_id].item()

    @modal.method()
    def causal_response_bank(
        self,
        prompt: str,
        features: list[tuple[int, int, int]],
        target_token: str,
        calibration_fractions: list[float] | None = None,
        validation_fractions: list[float] | None = None,
    ) -> dict:
        """Measure the same response tensor as the Qwen scorer on Gemma."""

        import torch

        calibration_fractions = list(
            CALIBRATION_FRACTIONS
            if calibration_fractions is None
            else calibration_fractions
        )
        validation_fractions = list(
            VALIDATION_FRACTIONS
            if validation_fractions is None
            else validation_fractions
        )
        if not features or len(set(features)) != len(features):
            raise ValueError("features must be a non-empty unique list")
        all_fractions = calibration_fractions + validation_fractions
        if (
            not calibration_fractions
            or not validation_fractions
            or any(not 0.0 < float(fraction) <= 1.0 for fraction in all_fractions)
            or set(calibration_fractions) & set(validation_fractions)
        ):
            raise ValueError(
                "calibration and validation fractions must be disjoint values in (0, 1]"
            )

        tokens, target_id = self._tokens_and_target(prompt, target_token)
        # nnsight's intervention hook mutates a traced activation tensor in
        # place; inference_mode forbids that operation, while no_grad retains
        # the same non-training numerics without marking tensors immutable.
        with torch.no_grad():
            base_logits, base_acts = self.model.get_activations(tokens)
            base_acts = self._normalize_acts(base_acts)
            base_output = self._target_log_odds(base_logits, target_id)
            base_values = [
                base_acts[layer, position, feature].float().item()
                for layer, feature, position in features
            ]
            directions = []
            for (layer, feature, _), base_value in zip(
                features, base_values, strict=True
            ):
                reference = base_acts[layer, :, feature].float().mean().item()
                directions.append(reference - base_value)

            def response_matrix(fraction: float) -> list[list[float]]:
                matrix: list[list[float]] = []
                for source, (
                    (layer, feature, position),
                    base_value,
                    direction,
                ) in enumerate(zip(features, base_values, directions, strict=True)):
                    delta = float(fraction) * direction
                    if abs(delta) <= 1e-12:
                        matrix.append([0.0] * (len(features) + 1))
                        continue
                    logits, acts = self.model.feature_intervention(
                        tokens,
                        [(layer, position, feature, base_value + delta)],
                    )
                    acts = self._normalize_acts(acts)
                    row = [
                        (
                            acts[dst_layer, dst_position, dst_feature].float().item()
                            - baseline
                        )
                        / delta
                        for (
                            dst_layer,
                            dst_feature,
                            dst_position,
                        ), baseline in zip(features, base_values, strict=True)
                    ]
                    row[source] = 0.0
                    row.append(
                        (self._target_log_odds(logits, target_id) - base_output) / delta
                    )
                    matrix.append(row)
                    del logits, acts
                return matrix

            calibration = [
                response_matrix(float(fraction))
                for fraction in calibration_fractions
            ]
            validation = [
                response_matrix(float(fraction)) for fraction in validation_fractions
            ]
            del base_logits, base_acts

        matrices = calibration + validation
        columns = len(features) + 1
        noise_scales: list[float] = []
        for destination in range(columns):
            variation = 0.0
            means: list[float] = []
            for source in range(len(features)):
                values = [matrix[source][destination] for matrix in matrices]
                mean = sum(values) / len(values)
                means.append(mean)
                variation += sum((value - mean) ** 2 for value in values)
            denominator = len(features) * max(len(matrices) - 1, 1)
            scale = math.sqrt(variation / denominator)
            if scale <= 1e-12:
                scale = math.sqrt(sum(value * value for value in means) / len(means))
            noise_scales.append(scale if scale > 1e-12 else 1.0)

        column_weights = [
            sum(matrix[source][-1] ** 2 for matrix in calibration) / len(calibration)
            for source in range(len(features))
        ] + [1.0]

        response_digest = hashlib.sha256(
            json.dumps(
                [calibration, validation, noise_scales, column_weights],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        bank_key = json.dumps(
            {
                "version": RESPONSE_BANK_VERSION,
                "model": MODEL_NAME,
                "transcoders": TRANSCODER_SET,
                "prompt": prompt,
                "target": target_token,
                "features": features,
                "calibration": calibration_fractions,
                "validation": validation_fractions,
                "response_digest": response_digest,
            },
            sort_keys=True,
        )
        return {
            "feature_ids": ["_".join(map(str, feature)) for feature in features],
            "calibration": calibration,
            "validation": validation,
            "noise_scales": noise_scales,
            "column_weights": column_weights,
            "metadata": {
                "bank_id": hashlib.sha256(bank_key.encode()).hexdigest()[:20],
                "response_digest": response_digest,
                "version": RESPONSE_BANK_VERSION,
                "model": MODEL_NAME,
                "transcoders": TRANSCODER_SET,
                "calibration_fractions": calibration_fractions,
                "validation_fractions": validation_fractions,
                "response": "local derivative toward across-position mean",
                "noise_scale": "pooled within-coordinate condition variation",
                "mediator_weight": "mean squared calibration output sensitivity",
            },
        }
