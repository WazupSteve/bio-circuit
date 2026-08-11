import hashlib
import json
import math
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-4B"
TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_FEATURE_NODES = 4096
RESPONSE_BANK_VERSION = "v1-local-derivative"
CALIBRATION_FRACTIONS = (0.25, 0.5)
VALIDATION_FRACTIONS = (0.75, 1.0)

app = modal.App("circuit-supernodes-attribution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "circuit-tracer==0.5.0",  # brings torch, transformer-lens, transformers
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
    max_containers=8,  # ceiling for RL-training fan-out; idle containers scale to 0
    scaledown_window=600,  # keep warm between reward batches (cold start is minutes)
)
class Attributor:
    @modal.enter()
    def load(self):
        import torch
        from circuit_tracer import ReplacementModel

        self.model = ReplacementModel.from_pretrained(
            MODEL_NAME, TRANSCODER_SET, dtype=torch.bfloat16
        )

    def _tokens_and_target(self, prompt: str, target_token: str):
        token_ids = self.model.tokenizer.encode(target_token, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"target token {target_token!r} encodes to nothing")
        return self.model.ensure_tokenized(prompt), token_ids[0]

    @staticmethod
    def _target_prob(logits, target_id: int) -> float:
        import torch

        final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
        return torch.softmax(final_logits, dim=-1)[target_id].item()

    @staticmethod
    def _target_log_odds(logits, target_id: int) -> float:
        """Stable target-vs-rest log odds, the response tensor's output unit."""

        import torch

        final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
        other_logits = torch.cat(
            (final_logits[:target_id], final_logits[target_id + 1 :])
        )
        return (final_logits[target_id] - torch.logsumexp(other_logits, dim=0)).item()

    @staticmethod
    def _normalize_acts(acts):
        return acts[:, 0] if acts.dim() == 4 else acts

    @staticmethod
    def _group_act(acts, members) -> float:
        layers, features, positions = zip(*members, strict=True)
        return acts[list(layers), list(positions), list(features)].sum().item()

    @staticmethod
    def _clamp_values(base_acts, members, ablation: str) -> list[float]:
        """The value each feature is clamped to.

        ``zero`` removes the feature outright. ``mean`` removes only its
        deviation from its own average activation over the prompt's positions,
        which reweights a group's members unequally — the property that makes it
        a genuinely different operator rather than a weaker version of the same
        one. Scoring uses ``mean`` while the agent's tool offers ``zero``, so a
        circuit fitted to what the agent measured has to generalise to be
        credited.
        """
        if ablation == "zero":
            return [0.0] * len(members)
        return [base_acts[layer, :, feat].mean().item() for layer, feat, _ in members]

    @modal.method()
    def intervene(
        self,
        prompt: str,
        groups: list[list[tuple[int, int, int]]],
        target_token: str,
        complement: list[tuple[int, int, int]] | None = None,
        ablation_indices: list[int] | None = None,
        ablation: str = "zero",
    ) -> dict:
        import torch

        # Attribution graphs index ctx_idx in BOS-prepended coordinates
        # (attribute() uses ensure_tokenized) — tokenize the same way here so
        # intervention positions line up.
        tokens, target_id = self._tokens_and_target(prompt, target_token)

        with torch.inference_mode():
            base_logits, base_acts = self.model.get_activations(tokens)
            base_acts = self._normalize_acts(base_acts)
            out = {
                "base_prob": self._target_prob(base_logits, target_id),
                "base_group_acts": [self._group_act(base_acts, g) for g in groups],
                "ablations": [],
                "complement_prob": None,
            }
            indices = (
                range(len(groups)) if ablation_indices is None else ablation_indices
            )
            for index in indices:
                g = groups[index]
                values = self._clamp_values(base_acts, g, ablation)
                interventions = [
                    (layer, pos, feat, v)
                    for (layer, feat, pos), v in zip(g, values, strict=True)
                ]
                logits, acts = self.model.feature_intervention(tokens, interventions)
                acts = self._normalize_acts(acts)
                out["ablations"].append(
                    {
                        "target_prob": self._target_prob(logits, target_id),
                        "group_acts": [
                            self._group_act(acts, group) for group in groups
                        ],
                    }
                )
                del logits, acts
            if complement:
                values = self._clamp_values(base_acts, complement, ablation)
                interventions = [
                    (layer, pos, feat, v)
                    for (layer, feat, pos), v in zip(complement, values, strict=True)
                ]
                logits, _ = self.model.feature_intervention(
                    tokens, interventions, return_activations=False
                )
                out["complement_prob"] = self._target_prob(logits, target_id)
                del logits
            del base_logits, base_acts
        return out

    @modal.method()
    def causal_response_bank(
        self,
        prompt: str,
        features: list[tuple[int, int, int]],
        target_token: str,
        calibration_fractions: list[float] | None = None,
        validation_fractions: list[float] | None = None,
    ) -> dict:
        """Measure a fixed full-universe causal response bank.

        Each feature is perturbed from its prompt-position activation toward its
        across-position mean.  Dividing every downstream change by the source
        perturbation produces a local causal derivative.  Calibration and
        validation use disjoint perturbation radii, so proposal-specific block
        values are fitted without reading the matrices that determine reward.
        """

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
        with torch.inference_mode():
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
                        for (dst_layer, dst_feature, dst_position), baseline in zip(
                            features, base_values, strict=True
                        )
                    ]
                    # The clamped source coordinate is an intervention artefact,
                    # not a causal self-edge available to the proposal language.
                    row[source] = 0.0
                    row.append(
                        (self._target_log_odds(logits, target_id) - base_output) / delta
                    )
                    matrix.append(row)
                    del logits, acts
                return matrix

            calibration = [
                response_matrix(float(fraction)) for fraction in calibration_fractions
            ]
            validation = [
                response_matrix(float(fraction)) for fraction in validation_fractions
            ]
            del base_logits, base_acts

        # Pooled within-coordinate variation across hidden intervention radii is
        # the frozen per-column noise estimate.  A deterministic column has zero
        # measured variation, so its RMS signal supplies the unique positive
        # scale available from the fixed bank; an all-zero column uses unit scale
        # and contributes no null energy.
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

        # First-order output propagation says mediator-column residuals should be
        # multiplied by (d output / d mediator)^2.  The calibration source row of
        # that mediator supplies this sensitivity without a hand-tuned weight.
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

    @modal.method()
    def attribute_prompt(self, prompt: str) -> dict | None:
        import torch
        from circuit_tracer import attribute
        from circuit_tracer.utils.create_graph_files import create_graph_files

        try:
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
            out_dir = Path("/tmp/graph_out")
            out_dir.mkdir(exist_ok=True)
            slug = "g"
            create_graph_files(
                graph_or_path=graph,
                slug=slug,
                output_path=str(out_dir),
                node_threshold=NODE_THRESHOLD,
                edge_threshold=EDGE_THRESHOLD,
            )
            data = json.loads((out_dir / f"{slug}.json").read_text())
            del graph
            torch.cuda.empty_cache()
            return data
        except Exception as exc:
            print(f"attribution failed for {prompt!r}: {exc}")
            return None


@app.local_entrypoint()
def main(prompt: str):
    graph = Attributor().attribute_prompt.remote(prompt)
    if graph is None:
        raise SystemExit("attribution failed")
    nodes = graph.get("nodes", [])
    print(f"{len(nodes)} nodes, {len(graph.get('links', []))} edges")
