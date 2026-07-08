"""Modal GPU service: attribution graphs for Qwen3-4B via circuit-tracer CLTs.

Loads Qwen3-4B as a circuit-tracer ReplacementModel with pretrained transcoders,
computes the attribution graph per prompt, prunes it with the paper's influence thresholds,
and returns slim graph JSON.

paper = Transformer Circuits : https://transformer-circuits.pub/2025/attribution-graphs/methods.html

test:
    uv run modal run modal_app/attribution.py --prompts-file modal_app/_prompts.json
"""

import json
from pathlib import Path

import modal

MODEL_NAME = "Qwen/Qwen3-4B"
# Pretrained per-layer transcoders for Qwen3-4B (160K features/layer, ReLU, bf16) by the circuit-tracer authors.
TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
NODE_THRESHOLD = 0.8  # keep nodes w/ cumulative influence >= 0.8 (paper default)
EDGE_THRESHOLD = 0.98  # then edges w/ cumulative score >= 0.98 (paper default)
MAX_FEATURE_NODES = 4096

app = modal.App("circuit-supernodes-attribution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "circuit-tracer",  # brings torch, transformer-lens, transformers
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
    max_containers=2,  # each container loads ~68GB of weights; don't fan out
)
class Attributor:
    @modal.enter()
    def load(self):
        import torch
        from circuit_tracer import ReplacementModel

        self.model = ReplacementModel.from_pretrained(
            MODEL_NAME, TRANSCODER_SET, dtype=torch.bfloat16
        )

    @modal.method()
    def intervene(
        self,
        prompt: str,
        groups: list[list[tuple[int, int, int]]],
        target_token: str,
    ) -> dict:
        """Ablate each group of features and measure causal propagation.

        groups: one list per proposed supernode of (layer, feature_idx, ctx_pos)
        triples (ctx_pos indexes the BOS-prepended token sequence, matching the
        attribution graph's ctx_idx)

        Returns base target-token probability and per-group summed activations,
        plus the same measurements under ablation of each group in turn.
        """
        import torch

        tid_seq = self.model.tokenizer.encode(target_token, add_special_tokens=False)
        if not tid_seq:
            raise ValueError(f"target token {target_token!r} encodes to nothing")
        tid = tid_seq[0]

        # Attribution graphs index ctx_idx in BOS-prepended coordinates
        # (attribute() uses ensure_tokenized) — tokenize the same way here so
        # intervention positions line up.
        tokens = self.model.ensure_tokenized(prompt)

        def target_prob(logits: torch.Tensor) -> float:
            vec = logits.reshape(-1, logits.shape[-1])[-1].float()
            return torch.softmax(vec, dim=-1)[tid].item()

        def norm_acts(acts: torch.Tensor) -> torch.Tensor:
            # [n_layers, batch, pos, d] -> [n_layers, pos, d]
            return acts[:, 0] if acts.dim() == 4 else acts

        def group_act(acts: torch.Tensor, members) -> float:
            return float(
                sum(acts[layer, pos, feat].item() for layer, feat, pos in members)
            )

        with torch.inference_mode():
            base_logits, base_acts = self.model.get_activations(tokens)
            base_acts = norm_acts(base_acts)
            out = {
                "base_prob": target_prob(base_logits),
                "base_group_acts": [group_act(base_acts, g) for g in groups],
                "ablations": [],
            }
            del base_logits, base_acts
            for g in groups:
                interventions = [(layer, pos, feat, 0.0) for layer, feat, pos in g]
                logits, acts = self.model.feature_intervention(tokens, interventions)
                acts = norm_acts(acts)
                out["ablations"].append(
                    {
                        "target_prob": target_prob(logits),
                        "group_acts": [group_act(acts, h) for h in groups],
                    }
                )
                del logits, acts
        torch.cuda.empty_cache()
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
        except Exception as e:  # return None so one bad prompt doesn't kill the batch
            print(f"attribution failed for {prompt!r}: {e}")
            return None


@app.local_entrypoint()
def main(prompts_file: str):
    prompts = json.loads(Path(prompts_file).read_text())
    attributor = Attributor()
    graphs = list(attributor.attribute_prompt.map(prompts, order_outputs=True))
    out = Path(prompts_file).parent / "_graphs.json"
    out.write_text(json.dumps(graphs))
    ok = sum(g is not None for g in graphs)
    print(f"{ok}/{len(prompts)} graphs computed -> {out}")
