"""GPU-side Circuit Tracer operations for persistent feature corruption.

This module owns model sampling, graph attribution, and feature-intervention
measurements. Deployment wiring lives in ``attribution_restoration.py``.
"""

import hashlib
import json
import types

import modal
from circuit_supernodes.runtime.ablation_contract import (
    ablation_result,
    evaluate_cohorts,
)
from circuit_supernodes.runtime.corruption import (
    CORRUPTION_VERSION,
    MAX_CORRUPTION_FACTOR,
    MAX_CORRUPTION_NODES,
    MIN_CORRUPTION_FACTOR,
    MIN_CORRUPTION_NODES,
    InterventionPayload,
    compose_multipliers,
)
from circuit_supernodes.runtime.graphs import MAX_FEATURE_NODES, parse_coordinate
from circuit_supernodes.runtime.model_ops import (
    normalize_activations,
    target_probability,
    tokens_and_target,
)

MODEL_NAME = "Qwen/Qwen3-4B"
TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_INTERVENTIONS = 60
MAX_ABS_WEIGHT = 8.0


# Keep the persistent-corruption backend isolated from legacy attribution
# deployments: Modal app names are account-global, so another worktree deploying
# the old class can otherwise remove restore_output while an eval is running.
app = modal.App("circuit-supernodes-persistent-restoration")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "circuit-tracer==0.5.0",
        "nnsight==0.6.1",
        "torch==2.8.0",
        "transformer-lens==2.16.1",
        # nnsight 0.6.1 imports two internals removed by cloudpickle 3.1.
        "cloudpickle==3.0.0",
        "huggingface_hub[hf_transfer]<1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("circuit_supernodes", "modal_app")
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
class RestorationAttributor:
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

    def _logits(self, tokens):
        batch = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
        return self.model._model(input_ids=batch).logits

    @modal.method()
    def sample_until_answer(self, request: dict) -> dict:
        """Use the shared bounded answer-aware sampler before attribution."""

        from circuit_supernodes.runtime.sampling import sample_until_answer

        return sample_until_answer(
            self.model,
            request,
            logits_fn=self._logits,
        )

    def _tokens_and_target(self, prompt: str, target_token: str):
        return tokens_and_target(
            self.model,
            prompt,
            target_token,
            require_single=True,
            require_round_trip=True,
        )

    def _intervention_input(self, tokens) -> str:
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
    def _normalize_acts(acts):
        return normalize_activations(acts)

    @modal.method()
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
                    "require_positive": True,
                },
            ).root
        )
        restorations = list(
            InterventionPayload.model_validate(
                restorations,
                context={
                    "label": "restoration",
                    "max_count": MAX_INTERVENTIONS,
                    "max_abs_factor": MAX_ABS_WEIGHT,
                },
            ).root
        )

        tokens, target_id = self._tokens_and_target(prompt, target_token)
        intervention_input = self._intervention_input(tokens)
        with torch.no_grad():
            standard_logits, base_acts = self.model.get_activations(tokens)
            base_acts = self._normalize_acts(base_acts)
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

    @modal.method()
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
                            "require_positive": True,
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
                        "max_abs_factor": MAX_ABS_WEIGHT,
                    },
                ).root
            )
            if any(factor == 0.0 for *_, factor in interventions):
                raise ValueError("ablation interventions must be nonzero")

            tokens, expected_id = self._tokens_and_target(prompt, expected_token)
            intervention_input = self._intervention_input(tokens)
            with torch.no_grad():
                clean_logits, base_acts = self.model.get_activations(tokens)
                base_acts = self._normalize_acts(base_acts)
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

    @staticmethod
    def _export_graph(graph, required_feature_ids: set[str] | None = None) -> dict:
        """Export the ordinary pruned graph while retaining required features.

        Required nodes use their real corrupted activation, influence and
        adjacency entries. We merely pin each node and its strongest incoming
        and outgoing graph edges through the same frontend exporter.
        """

        import torch
        from circuit_tracer.graph import prune_graph
        from circuit_tracer.utils.create_graph_files import (
            build_model,
            create_nodes,
            create_used_nodes_and_edges,
        )
        from transformers import AutoTokenizer

        slug = "g"
        graph.to("cuda" if torch.cuda.is_available() else "cpu")
        node_mask, edge_mask, cumulative_scores = (
            value.cpu() for value in prune_graph(graph, NODE_THRESHOLD, EDGE_THRESHOLD)
        )
        graph.to("cpu")

        required_feature_ids = required_feature_ids or set()
        found: set[str] = set()
        adjacency = graph.adjacency_matrix
        for node_index, selected_index in enumerate(graph.selected_features.tolist()):
            layer, position, feature = graph.active_features[selected_index].tolist()
            node_id = f"{layer}_{feature}_{position}"
            if node_id not in required_feature_ids:
                continue
            found.add(node_id)
            node_mask[node_index] = True
            for scores, incoming in (
                (adjacency[node_index].abs(), True),
                (adjacency[:, node_index].abs(), False),
            ):
                nonzero = scores.nonzero(as_tuple=False).flatten()
                if not len(nonzero):
                    continue
                keep = nonzero[
                    torch.topk(scores[nonzero], min(5, len(nonzero))).indices
                ]
                node_mask[keep] = True
                if incoming:
                    edge_mask[node_index, keep] = True
                else:
                    edge_mask[keep, node_index] = True
        if missing := required_feature_ids - found:
            raise ValueError(
                f"required corrupted features were not attributed: {sorted(missing)}"
            )

        tokenizer = AutoTokenizer.from_pretrained(graph.cfg.tokenizer_name)
        nodes = create_nodes(graph, node_mask, tokenizer, cumulative_scores)
        used_nodes, used_edges = create_used_nodes_and_edges(graph, nodes, edge_mask)
        model = build_model(
            graph,
            used_nodes,
            used_edges,
            slug,
            graph.scan,
            NODE_THRESHOLD,
            tokenizer,
        )
        return model.model_dump(mode="json")

    def _corrupted_attribution_context(self, inputs, interventions):
        """Build circuit-tracer's phase-0 context from a corrupted execution."""

        import torch
        from circuit_tracer.attribution.context_nnsight import AttributionContext
        from nnsight import save

        model = self.model
        tokens = (
            model.ensure_tokenized(inputs)
            if isinstance(inputs, str)
            else inputs.squeeze()
        )
        intervention_layers = {int(item[0]) for item in interventions}
        activation_matrix, activation_fn = model.get_activation_fn()

        with model.trace() as tracer:
            activation_barrier = tracer.barrier(2)
            with tracer.invoke(tokens):
                activation_fn(
                    barrier=activation_barrier,
                    barrier_layers=intervention_layers,
                    activation_layers=sorted(intervention_layers),
                )
            with tracer.invoke():
                # A shape-only tensor avoids _perform_feature_intervention trying
                # to tokenize an already-tokenized tensor. It is otherwise unused
                # when constrained_layers is None.
                shape_stub = torch.empty((1, tokens.numel(), 1), device=model.device)
                model._perform_feature_intervention(
                    tokens,
                    interventions,
                    activation_matrix,
                    shape_stub,
                    activation_barrier,
                    None,
                    constrained_layers=None,
                )
            with tracer.invoke():
                mlp_inputs, mlp_outputs = [], []
                for feature_input_loc, feature_output_loc in zip(
                    model.feature_input_locs,
                    model.feature_output_locs,
                    strict=True,
                ):
                    mlp_inputs.append(feature_input_loc.output)
                    output = feature_output_loc.output
                    if output.ndim == 2:
                        output = output.unsqueeze(0)
                    mlp_outputs.append(output)
                mlp_inputs = save(torch.cat(mlp_inputs, dim=0))
                mlp_outputs = save(torch.cat(mlp_outputs, dim=0))
                logits = save(model.output.logits)

        components = model.transcoders.compute_attribution_components(
            mlp_inputs, model.zero_positions
        )
        sparse = components["activation_matrix"].coalesce()
        indices = sparse.indices().clone()
        values = sparse.values().clone()
        reconstruction = components["reconstruction"].clone()
        transcoders = (
            model.transcoders._module
            if hasattr(model.transcoders, "_module")
            else model.transcoders
        )

        # The encoder sees the natural pre-intervention value. Replace it with
        # the value actually written into the residual stream, and make its
        # scaled decoder vector and reconstruction agree with that execution.
        for layer, position, feature, desired_value in interventions:
            matches = (
                (
                    (indices[0] == layer)
                    & (indices[1] == position)
                    & (indices[2] == feature)
                )
                .nonzero(as_tuple=False)
                .flatten()
            )
            decoder = transcoders._get_decoder_vectors(
                int(layer), torch.tensor([feature], device=values.device)
            )[0]
            if len(matches) == 1:
                index = int(matches[0])
                delta = float(desired_value) - values[index].float()
                values[index] = float(desired_value)
            elif len(matches) == 0:
                coordinate = torch.tensor(
                    [[layer], [position], [feature]],
                    dtype=indices.dtype,
                    device=indices.device,
                )
                indices = torch.cat((indices, coordinate), dim=1)
                values = torch.cat(
                    (
                        values,
                        torch.tensor(
                            [desired_value], dtype=values.dtype, device=values.device
                        ),
                    )
                )
                delta = torch.tensor(
                    float(desired_value), dtype=values.dtype, device=values.device
                )
            else:
                raise ValueError("duplicate sparse attribution coordinate")
            reconstruction[layer, position] += decoder * delta

        activation_matrix = torch.sparse_coo_tensor(
            indices, values, sparse.shape, device=values.device
        ).coalesce()
        (
            _,
            _,
            _,
            decoder_vecs,
            encoder_to_decoder_map,
        ) = transcoders.select_decoder_vectors(activation_matrix)
        final_indices = activation_matrix.indices()
        encoder_vecs = torch.cat(
            [
                transcoders[layer].W_enc[final_indices[2, final_indices[0] == layer]]
                for layer in final_indices[0].unique(sorted=True).tolist()
            ]
        )
        error_vectors = mlp_outputs - reconstruction
        error_vectors[:, model.zero_positions] = 0
        token_vectors = model.embed_weight[tokens].detach()
        return AttributionContext(
            activation_matrix=activation_matrix,
            logits=logits,
            error_vectors=error_vectors,
            token_vectors=token_vectors,
            decoder_vecs=decoder_vecs,
            encoder_vecs=encoder_vecs,
            encoder_to_decoder_map=encoder_to_decoder_map,
            decoder_locations=activation_matrix.indices()[:2],
        )

    def _configure_corrupted_gradient_flow(
        self, tracer, interventions, original_configure, n_positions
    ):
        """Replay corruption in circuit-tracer's phase-1 attribution forward."""

        import torch

        model = self.model
        transcoders = (
            model.transcoders._module
            if hasattr(model.transcoders, "_module")
            else model.transcoders
        )
        intervention_layers = {int(item[0]) for item in interventions}
        activations = [None] * model.cfg.n_layers
        activation_barrier = tracer.barrier(2)
        with tracer.invoke():
            for layer in sorted(intervention_layers):
                acts = transcoders.encode_layer(
                    model.get_feature_input_loc(layer).output, layer
                ).detach()[0]
                acts[model.zero_positions] = 0
                activations[layer] = acts
                activation_barrier()
        with tracer.invoke():
            shape_stub = torch.empty((1, n_positions, 1), device=model.device)
            # Only size(1) is read, and layer deltas derive their actual shape
            # from the activation proxies above.
            model._perform_feature_intervention(
                "",
                interventions,
                activations,
                shape_stub,
                activation_barrier,
                None,
                constrained_layers=None,
            )
        original_configure(tracer)

    def _attribute_graph(self, prompt: str, interventions=None):
        """Attribute a clean or persistently corrupted execution."""

        import torch
        from circuit_tracer import attribute

        interventions = list(interventions or [])
        original_setup = self.model.setup_attribution
        original_configure = self.model.configure_gradient_flow
        if interventions:
            owner = self
            n_positions = int(self.model.ensure_tokenized(prompt).numel())

            def setup(model, inputs):
                return owner._corrupted_attribution_context(inputs, interventions)

            def configure(model, tracer):
                return owner._configure_corrupted_gradient_flow(
                    tracer, interventions, original_configure, n_positions
                )

            self.model.setup_attribution = types.MethodType(setup, self.model)
            self.model.configure_gradient_flow = types.MethodType(configure, self.model)
        try:
            graph = attribute(
                prompt=prompt,
                model=self.model,
                max_n_logits=5,
                desired_logit_prob=0.95,
                max_feature_nodes=MAX_FEATURE_NODES,
                # NNSight retains a much larger autograd graph than the
                # TransformerLens backend. Reused generator containers need
                # additional headroom after corruption screening.
                batch_size=8,
                offload=None,
                verbose=True,
            )
            required = {
                f"{int(layer)}_{int(feature)}_{int(position)}"
                for layer, position, feature, _ in interventions
            }
            return self._export_graph(graph, required)
        finally:
            self.model.setup_attribution = original_setup
            self.model.configure_gradient_flow = original_configure
            if "graph" in locals():
                del graph
            torch.cuda.empty_cache()

    @staticmethod
    def _feature_nodes(graph: dict) -> list[dict]:
        nodes = []
        for node in graph.get("nodes", []):
            node_id = str(node.get("node_id", node.get("id", "")))
            try:
                parse_coordinate(node_id)
            except Exception:
                continue
            if float(node.get("activation", 0.0) or 0.0) > 0:
                nodes.append(node)
        return sorted(nodes, key=lambda node: float(node.get("influence", 1.0)))

    @classmethod
    def _visible_feature_nodes(cls, graph: dict, limit: int = 60) -> list[dict]:
        """Mirror TracerGraphPayload's cumulative-influence conversion."""

        influence_deltas = []
        previous = 0.0
        ordered = sorted(
            graph.get("nodes", []),
            key=lambda node: float(node.get("influence", 0.0) or 0.0),
        )
        feature_objects = {id(node): node for node in cls._feature_nodes(graph)}
        for node in ordered:
            influence = float(node.get("influence", 0.0) or 0.0)
            influence_delta = max(influence - previous, 0.0)
            previous = influence
            if id(node) in feature_objects:
                node_id = str(node.get("node_id", node.get("id", "")))
                influence_deltas.append((influence_delta, node_id, node))
        influence_deltas.sort(key=lambda item: (-item[0], item[1]))
        return [node for _, _, node in influence_deltas[:limit]]

    def _corrupt_with_stable_spec(
        self,
        prompt: str,
        expected_target_token: str,
        stable_spec: list[tuple[int, int, float]],
        *,
        export_graph: bool,
    ) -> dict:
        """Apply one stable corruption spec at its top position in this prompt."""

        clean_graph = self._attribute_graph(prompt)
        candidates = self._visible_feature_nodes(clean_graph, MAX_FEATURE_NODES)
        by_stable: dict[tuple[int, int], dict] = {}
        for node in candidates:
            layer, feature, _ = parse_coordinate(
                str(node.get("node_id", node.get("id", "")))
            )
            by_stable.setdefault((layer, feature), node)

        selected = []
        for layer, feature, factor in stable_spec:
            node = by_stable.get((int(layer), int(feature)))
            if node is None:
                continue
            _, _, position = parse_coordinate(
                str(node.get("node_id", node.get("id", "")))
            )
            selected.append((int(layer), int(feature), position, float(factor)))
        if len(selected) < MIN_CORRUPTION_NODES:
            return {"rejected": "too few stable corruption features in prompt"}
        selected = selected[:MAX_CORRUPTION_NODES]

        return self._corrupt_with_coordinate_spec(
            prompt, expected_target_token, selected, export_graph=export_graph
        )

    def _corrupt_with_coordinate_spec(
        self,
        prompt: str,
        expected_target_token: str,
        coordinate_spec: list[tuple[int, int, int, float]],
        *,
        export_graph: bool,
    ) -> dict:
        """Apply corruption coordinates already resolved from a stored clean graph."""

        import torch

        selected = [
            (int(layer), int(feature), int(position), float(factor))
            for layer, feature, position, factor in coordinate_spec
        ][:MAX_CORRUPTION_NODES]
        if len(selected) < MIN_CORRUPTION_NODES:
            return {"rejected": "too few corruption coordinates in prompt"}

        tokens = self.model.ensure_tokenized(prompt)
        intervention_input = self._intervention_input(tokens)
        with torch.no_grad():
            clean_logits, clean_acts = self.model.get_activations(tokens)
            clean_acts = self._normalize_acts(clean_acts)
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
            "corruption_version": CORRUPTION_VERSION,
        }
        if not changed or not export_graph:
            return result
        graph = self._attribute_graph(prompt, absolute)
        exported = {
            str(node.get("node_id", node.get("id", "")))
            for node in self._feature_nodes(graph)
        }
        hidden = {
            f"{layer}_{feature}_{position}" for layer, feature, position, _ in selected
        }
        if not hidden.issubset(exported):
            return {"rejected": "corruption features absent from corrupted graph"}
        digest = hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return {**result, "graph": graph, "corruption_id": digest}

    @modal.method()
    def test_corruption_coordinates(
        self,
        prompt: str,
        expected_target_token: str,
        coordinate_spec: list[tuple[int, int, int, float]],
    ) -> dict:
        return self._corrupt_with_coordinate_spec(
            prompt, expected_target_token, coordinate_spec, export_graph=False
        )

    @modal.method()
    def build_corrupted_sample_from_coordinates(
        self,
        prompt: str,
        expected_target_token: str,
        coordinate_spec: list[tuple[int, int, int, float]],
    ) -> dict:
        return self._corrupt_with_coordinate_spec(
            prompt, expected_target_token, coordinate_spec, export_graph=True
        )

    @modal.method()
    def build_corrupted_sample_from_spec(
        self,
        prompt: str,
        expected_target_token: str,
        stable_spec: list[tuple[int, int, float]],
    ) -> dict:
        return self._corrupt_with_stable_spec(
            prompt, expected_target_token, stable_spec, export_graph=True
        )

    @modal.method()
    def attribute_prompt(self, prompt: str) -> dict | None:
        try:
            return self._attribute_graph(prompt)
        except Exception as exc:
            print(f"attribution failed for {prompt!r}: {exc}")
            return None
