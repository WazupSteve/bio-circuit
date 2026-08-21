"""Modal deployment configuration for the unified backend."""

import modal

MODAL_APP = "circuit-supernodes"
MODAL_CLS = "Attributor"

app = modal.App(MODAL_APP)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "circuit-tracer==0.5.0",
        "nnsight==0.6.1",
        "torch==2.8.0",
        "transformer-lens==2.16.1",
        "cloudpickle==3.0.0",
        "huggingface_hub[hf_transfer]<1",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("circuit_supernodes", "modal_app")
)
hf_cache = modal.Volume.from_name("circuit-supernodes-hf-cache", create_if_missing=True)
