import json
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-4B"
TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98
MAX_FEATURE_NODES = 4096

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
    def _normalize_acts(acts):
        return acts[:, 0] if acts.dim() == 4 else acts

    @staticmethod
    def _group_act(acts, members) -> float:
        layers, features, positions = zip(*members, strict=True)
        return acts[list(layers), list(positions), list(features)].sum().item()

    @modal.method()
    def intervene(
        self,
        prompt: str,
        groups: list[list[tuple[int, int, int]]],
        target_token: str,
        complement: list[tuple[int, int, int]] | None = None,
        ablation_indices: list[int] | None = None,
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
            del base_logits, base_acts
            indices = (
                range(len(groups)) if ablation_indices is None else ablation_indices
            )
            for index in indices:
                g = groups[index]
                interventions = [(layer, pos, feat, 0.0) for layer, feat, pos in g]
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
                interventions = [
                    (layer, pos, feat, 0.0) for layer, feat, pos in complement
                ]
                logits, _ = self.model.feature_intervention(
                    tokens, interventions, return_activations=False
                )
                out["complement_prob"] = self._target_prob(logits, target_id)
                del logits
        return out

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
