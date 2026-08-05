"""Generic Modal profiler for aligned feature suppression comparisons."""

import hashlib
import json

import modal

APP_NAME = "circuit-feature-suppression"
CLASS_NAME = "SuppressionProfiler"

app = modal.App(APP_NAME)

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
    max_containers=2,
    scaledown_window=300,
)
class SuppressionProfiler:
    model_name: str = modal.parameter()
    transcoder_set: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from circuit_tracer import ReplacementModel

        self.model = ReplacementModel.from_pretrained(
            self.model_name,
            self.transcoder_set,
            backend="nnsight",
            dtype=torch.bfloat16,
        )

    @staticmethod
    def _normalize_acts(acts):
        return acts[:, 0] if acts.dim() == 4 else acts

    @staticmethod
    def _target_log_odds(logits, target_id: int) -> float:
        import torch

        final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
        other_logits = torch.cat(
            (final_logits[:target_id], final_logits[target_id + 1 :])
        )
        return (final_logits[target_id] - torch.logsumexp(other_logits, dim=0)).item()

    @modal.method()
    def profile_features(
        self,
        prompt: str,
        features: list[tuple[int, int, int]],
        target_token: str,
    ) -> dict:
        """Measure each feature's realized target support under a mean clamp."""

        import torch

        if not features or len(set(features)) != len(features):
            raise ValueError("features must be a non-empty unique list")
        target_ids = self.model.tokenizer.encode(
            target_token, add_special_tokens=False
        )
        if not target_ids:
            raise ValueError(f"target token {target_token!r} encodes to nothing")
        tokens = self.model.ensure_tokenized(prompt)
        token_values = tokens.detach().cpu().reshape(-1).tolist()
        token_fingerprint = hashlib.sha256(
            json.dumps(
                [token_values, target_ids], separators=(",", ":")
            ).encode()
        ).hexdigest()

        with torch.no_grad():
            base_logits, base_acts = self.model.get_activations(tokens)
            base_acts = self._normalize_acts(base_acts)
            base_output = self._target_log_odds(base_logits, target_ids[0])
            measurements = []
            for layer, feature, position in features:
                if not (
                    0 <= layer < base_acts.shape[0]
                    and 0 <= position < base_acts.shape[1]
                    and 0 <= feature < base_acts.shape[2]
                ):
                    raise ValueError(
                        f"feature coordinate {(layer, feature, position)!r} "
                        f"is outside activation shape {tuple(base_acts.shape)!r}"
                    )
                activation = base_acts[layer, position, feature].float().item()
                mean_activation = base_acts[layer, :, feature].float().mean().item()
                logits, _ = self.model.feature_intervention(
                    tokens,
                    [(layer, position, feature, mean_activation)],
                    return_activations=False,
                )
                changed_output = self._target_log_odds(logits, target_ids[0])
                measurements.append(
                    {
                        "feature_id": f"{layer}_{feature}_{position}",
                        "feature_key": f"{layer}_{feature}",
                        "activation": activation,
                        "mean_activation": mean_activation,
                        "activation_excess": activation - mean_activation,
                        "base_target_log_odds": base_output,
                        "mean_clamp_target_log_odds": changed_output,
                        "mean_clamp_support_log_odds": base_output - changed_output,
                    }
                )
                del logits
            del base_logits, base_acts

        return {
            "model": self.model_name,
            "transcoders": self.transcoder_set,
            "target_token": target_token,
            "target_token_id": target_ids[0],
            "token_fingerprint": token_fingerprint,
            "measurements": measurements,
        }
