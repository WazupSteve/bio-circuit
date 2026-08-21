"""Gemma-2-2B circuit tracing and functional-replacement grading."""

import modal
from modal_app.attribution_service import create_attributor_service

MODEL_NAME = "unsloth/gemma-2-2b"
TRANSCODER_SET = "mntss/gemma-scope-transcoders"
NODE_THRESHOLD = 0.8
EDGE_THRESHOLD = 0.98

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
    .add_local_python_source("circuit_supernodes", "modal_app")
)

hf_cache = modal.Volume.from_name("circuit-supernodes-hf-cache", create_if_missing=True)


def _gemma_logits(model, tokens):
    batch = tokens.unsqueeze(0) if tokens.dim() == 1 else tokens
    return model._model(input_ids=batch).logits


GemmaAttributor = create_attributor_service(
    app=app,
    class_name="GemmaAttributor",
    image=image,
    model_name=MODEL_NAME,
    transcoder_set=TRANSCODER_SET,
    node_threshold=NODE_THRESHOLD,
    edge_threshold=EDGE_THRESHOLD,
    volume=hf_cache,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=4,
    scaledown_window=300,
    model_options={"backend": "nnsight"},
    logits_adapter=_gemma_logits,
    require_single_token=True,
    require_round_trip=True,
    validate_activation_shape=True,
)
