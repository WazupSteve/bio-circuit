import hashlib
import json
import random
import types

import modal
from circuit_supernodes.interventions import (
    CORRUPTION_VERSION,
    MAX_ABS_WEIGHT,
    MAX_CORRUPTION_FACTOR,
    MAX_CORRUPTION_NODES,
    MAX_INTERVENTIONS,
    MIN_CORRUPTION_FACTOR,
    MIN_CORRUPTION_NODES,
    compose_multipliers,
    coordinate_from_feature_id,
    validate_interventions,
)

MODEL_NAME = "Qwen/Qwen3-4B"
TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_FEATURE_NODES = 4096
CORRUPTION_MIN_NODES = MIN_CORRUPTION_NODES
CORRUPTION_MAX_NODES = MAX_CORRUPTION_NODES
CORRUPTION_MIN_LOG2_MAGNITUDE = 1.0
CORRUPTION_MAX_LOG2_MAGNITUDE = 3.0
CORRUPTION_CANDIDATES = 60
CORRUPTION_ATTEMPTS = 32
BLANKET_RESTORATION_WEIGHTS = (0.25, 0.353553, 0.176777, 0.5, 0.125)

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
    .add_local_python_source("circuit_supernodes")
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
            MODEL_NAME,
            TRANSCODER_SET,
            backend="nnsight",
            dtype=torch.bfloat16,
        )

    def _tokens_and_target(self, prompt: str, target_token: str):
        token_ids = self.model.tokenizer.encode(target_token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"target token {target_token!r} must encode to exactly one token"
            )
        if self.model.tokenizer.decode(token_ids) != target_token:
            raise ValueError(f"target token {target_token!r} does not round-trip")
        return self.model.ensure_tokenized(prompt), token_ids[0]

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
    def _target_prob(logits, target_id: int) -> float:
        import torch

        final_logits = logits.reshape(-1, logits.shape[-1])[-1].float()
        return torch.softmax(final_logits, dim=-1)[target_id].item()

    @staticmethod
    def _normalize_acts(acts):
        return acts[:, 0] if acts.dim() == 4 else acts

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

        corruption = validate_interventions(
            corruption,
            label="corruption",
            require_nonempty=True,
            min_count=MIN_CORRUPTION_NODES,
            max_count=MAX_CORRUPTION_NODES,
            min_factor=MIN_CORRUPTION_FACTOR,
            max_abs_factor=MAX_CORRUPTION_FACTOR,
            require_positive=True,
        )
        restorations = validate_interventions(
            restorations,
            label="restoration",
            max_count=MAX_INTERVENTIONS,
            max_abs_factor=MAX_ABS_WEIGHT,
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
            standard_probs = standard_log_probs.exp()

            def standard_kl(candidate) -> float:
                candidate_log_probs = torch.log_softmax(candidate, dim=-1)
                return (
                    standard_probs * (standard_log_probs - candidate_log_probs)
                ).sum().item()

            standard_top_id = int(standard_final.argmax().item())
            corrupted_top_id = int(corrupted_final.argmax().item())
            restored_top_id = int(restored_final.argmax().item())
            out = {
                "target_token": target_token,
                "standard_target_probability": self._target_prob(
                    standard_logits, target_id
                ),
                "corrupted_target_probability": self._target_prob(
                    corrupted_logits, target_id
                ),
                "restored_target_probability": self._target_prob(
                    restored_logits, target_id
                ),
                "corrupted_standard_kl": standard_kl(corrupted_final),
                "restored_standard_kl": standard_kl(restored_final),
                "standard_top_token": self.model.tokenizer.decode([standard_top_id]),
                "corrupted_top_token": self.model.tokenizer.decode(
                    [corrupted_top_id]
                ),
                "restored_top_token": self.model.tokenizer.decode([restored_top_id]),
                "standard_target_is_top": standard_top_id == target_id,
                "corruption_changed_top_token": corrupted_top_id != standard_top_id,
                "restored_target_is_top": restored_top_id == target_id,
            }
            del standard_logits, corrupted_logits, restored_logits, base_acts
        return out

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
            value.cpu()
            for value in prune_graph(graph, NODE_THRESHOLD, EDGE_THRESHOLD)
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
        return json.loads(model.model_dump_json())

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
                (indices[0] == layer)
                & (indices[1] == position)
                & (indices[2] == feature)
            ).nonzero(as_tuple=False).flatten()
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
            shape_stub = torch.empty(
                (1, n_positions, 1), device=model.device
            )
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
                coordinate_from_feature_id(node_id)
            except ValueError:
                continue
            if float(node.get("activation", 0.0) or 0.0) > 0:
                nodes.append(node)
        return sorted(nodes, key=lambda node: float(node.get("influence", 1.0)))

    @classmethod
    def _visible_feature_nodes(cls, graph: dict, limit: int = 60) -> list[dict]:
        """Mirror graphs.slim_graph's cumulative-influence conversion."""

        shares = []
        previous = 0.0
        ordered = sorted(
            graph.get("nodes", []),
            key=lambda node: float(node.get("influence", 0.0) or 0.0),
        )
        feature_objects = {id(node): node for node in cls._feature_nodes(graph)}
        for node in ordered:
            influence = float(node.get("influence", 0.0) or 0.0)
            share = max(influence - previous, 0.0)
            previous = influence
            if id(node) in feature_objects:
                node_id = str(node.get("node_id", node.get("id", "")))
                shares.append((share, node_id, node))
        shares.sort(key=lambda item: (-item[0], item[1]))
        return [node for _, _, node in shares[:limit]]

    @modal.method()
    def build_corrupted_sample(
        self, prompt: str, expected_target: str, seed: int
    ) -> dict | None:
        """Create one accepted graph and its compact, hidden replay payload."""

        import torch

        try:
            clean_graph = self._attribute_graph(prompt)
            tokens = self.model.ensure_tokenized(prompt)
            intervention_input = self._intervention_input(tokens)
            clean_logits, clean_acts = self.model.get_activations(tokens)
            clean_acts = self._normalize_acts(clean_acts)
            clean_final = clean_logits.reshape(-1, clean_logits.shape[-1])[-1]
            target_id = int(clean_final.argmax().item())
            target_token = self.model.tokenizer.decode([target_id])
            if not target_token.strip() or not expected_target.strip().startswith(
                target_token.strip()
            ):
                return {
                    "rejected": "clean target mismatch",
                    "target_token": target_token,
                }

            candidates = self._visible_feature_nodes(
                clean_graph, CORRUPTION_CANDIDATES
            )
            if len(candidates) < CORRUPTION_MIN_NODES:
                return {"rejected": "too few clean repository nodes"}
            rng = random.Random(int(seed))
            clean_probs = clean_final.float().softmax(dim=-1)
            lowest_target_prob = float(clean_probs[target_id].item())
            strongest_competitor_prob = float(
                clean_probs.masked_fill(
                    torch.arange(clean_probs.numel(), device=clean_probs.device)
                    == target_id,
                    float("-inf"),
                )
                .max()
                .item()
            )

            def blanket_restores(graph, hidden_factors, base_activations):
                hidden_ids = {
                    f"{layer}_{feature}_{position}"
                    for layer, feature, position in hidden_factors
                }
                ranked_visible = self._visible_feature_nodes(
                    graph, CORRUPTION_CANDIDATES
                )
                visible_ids = list(hidden_ids)
                visible_ids.extend(
                    str(node.get("node_id", node.get("id", "")))
                    for node in ranked_visible
                    if str(node.get("node_id", node.get("id", "")))
                    not in hidden_ids
                )
                visible_coordinates = [
                    coordinate_from_feature_id(node_id)
                    for node_id in visible_ids[:CORRUPTION_CANDIDATES]
                ]
                for weight in BLANKET_RESTORATION_WEIGHTS:
                    factors = dict(hidden_factors)
                    for coordinate in visible_coordinates:
                        factors[coordinate] = factors.get(coordinate, 1.0) * weight
                    blanket_absolute = [
                        (
                            layer,
                            position,
                            feature,
                            base_activations[layer, position, feature].float().item()
                            * factor,
                        )
                        for (layer, feature, position), factor in sorted(
                            factors.items()
                        )
                    ]
                    blanket_logits, _ = self.model.feature_intervention(
                        intervention_input,
                        blanket_absolute,
                        freeze_attention=False,
                        return_activations=False,
                    )
                    blanket_final = blanket_logits.reshape(
                        -1, blanket_logits.shape[-1]
                    )[-1]
                    solved = int(blanket_final.argmax().item()) == target_id
                    del blanket_logits, blanket_final
                    if solved:
                        return True
                return False

            blanket_rejections = 0
            for attempt in range(CORRUPTION_ATTEMPTS):
                count = rng.randint(
                    CORRUPTION_MIN_NODES,
                    min(CORRUPTION_MAX_NODES, len(candidates)),
                )
                chosen = rng.sample(candidates, count)
                corruption = []
                absolute = []
                for node in chosen:
                    layer, feature, position = map(
                        int, str(node.get("node_id", node.get("id"))).split("_")
                    )
                    magnitude = rng.uniform(
                        CORRUPTION_MIN_LOG2_MAGNITUDE,
                        CORRUPTION_MAX_LOG2_MAGNITUDE,
                    )
                    factor = round(2**magnitude, 6)
                    base = clean_acts[layer, position, feature].float().item()
                    corruption.append((layer, feature, position, factor))
                    absolute.append((layer, position, feature, base * factor))

                corrupted_logits, _ = self.model.feature_intervention(
                    intervention_input,
                    absolute,
                    freeze_attention=False,
                    return_activations=False,
                )
                corrupted_final = corrupted_logits.reshape(
                    -1, corrupted_logits.shape[-1]
                )[-1]
                corrupted_probs = corrupted_final.float().softmax(dim=-1)
                lowest_target_prob = min(
                    lowest_target_prob, float(corrupted_probs[target_id].item())
                )
                competitor_prob = corrupted_probs.masked_fill(
                    torch.arange(
                        corrupted_probs.numel(), device=corrupted_probs.device
                    )
                    == target_id,
                    float("-inf"),
                ).max()
                strongest_competitor_prob = max(
                    strongest_competitor_prob, float(competitor_prob.item())
                )
                changed_top = int(corrupted_final.argmax().item()) != target_id
                del corrupted_logits, corrupted_final, corrupted_probs, competitor_prob
                if not changed_top:
                    continue
                hidden_factors = {item[:3]: item[3] for item in corruption}
                # Most shortcut candidates can be rejected against the clean
                # graph's visible slice before paying for corrupted attribution.
                # The exact corrupted slice is checked again below.
                if blanket_restores(clean_graph, hidden_factors, clean_acts):
                    blanket_rejections += 1
                    continue
                torch.cuda.empty_cache()
                try:
                    graph = self._attribute_graph(prompt, absolute)
                except (RuntimeError, ValueError) as exc:
                    print(f"corrupted attribution attempt {attempt} rejected: {exc}")
                    continue
                displayed = {
                    str(node.get("node_id", node.get("id", "")))
                    for node in self._feature_nodes(graph)
                }
                hidden_ids = {
                    f"{layer}_{feature}_{position}"
                    for layer, feature, position, _ in corruption
                }
                if not hidden_ids.issubset(displayed):
                    continue

                # Reject tasks solved by scaling every visible feature with one
                # generic inverse factor. This known shortcut belongs in dataset
                # admission rather than in the rollout reward contract.
                if blanket_restores(graph, hidden_factors, clean_acts):
                    blanket_rejections += 1
                    continue
                digest = hashlib.sha256(
                    json.dumps(corruption, separators=(",", ":")).encode()
                ).hexdigest()[:16]
                del clean_logits, clean_acts
                return {
                    "graph": graph,
                    "target_token": target_token,
                    "corruption": corruption,
                    "corruption_id": digest,
                    "corruption_version": CORRUPTION_VERSION,
                    "corruption_attempt": attempt,
                }
            return {
                "rejected": (
                    "no admissible corruption found "
                    f"(blanket shortcut rejected {blanket_rejections}, "
                    f"lowest target p={lowest_target_prob:.4g}, "
                    f"largest competitor p={strongest_competitor_prob:.4g})"
                )
            }
        except Exception as exc:
            print(f"corrupted sample failed for {prompt!r}: {exc}")
            return None

    @modal.method()
    def attribute_prompt(self, prompt: str) -> dict | None:
        try:
            return self._attribute_graph(prompt)
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
