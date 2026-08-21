import json

import modal
from modal_app.attribution_service import create_attributor_service

MODEL_NAME = "Qwen/Qwen3-4B"
TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98

app = modal.App("circuit-supernodes-attribution")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "circuit-tracer==0.5.0",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("circuit_supernodes", "modal_app")
)

hf_cache = modal.Volume.from_name("circuit-supernodes-hf-cache", create_if_missing=True)


Attributor = create_attributor_service(
    app=app,
    class_name="Attributor",
    image=image,
    model_name=MODEL_NAME,
    transcoder_set=TRANSCODER_SET,
    node_threshold=NODE_THRESHOLD,
    edge_threshold=EDGE_THRESHOLD,
    volume=hf_cache,
    max_containers=8,
    scaledown_window=600,
    forgive_attribution_errors=True,
)


@app.local_entrypoint()
def main(prompt: str, answers: str = "", max_new_tokens: int = 50):
    if answers:
        result = Attributor().sample_until_answer.remote(
            {
                "prompt": prompt,
                "possible_answers": answers.split(","),
                "max_new_tokens": max_new_tokens,
            }
        )
        print(json.dumps(result, indent=2))
        return
    graph = Attributor().attribute_prompt.remote(prompt)
    if graph is None:
        raise SystemExit("attribution failed")
    print(f"{len(graph.get('nodes', []))} nodes, {len(graph.get('links', []))} edges")
