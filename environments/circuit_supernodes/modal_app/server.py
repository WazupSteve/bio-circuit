"""Self-hosted subject server: run on any GPU box, reach it via SSH tunnel.

    python -m modal_app.server --port 8000

Each ``POST /<method>`` takes a JSON list of positional args and returns the
method's JSON result. GPU work is serialised behind one lock.
"""

import threading
from typing import Annotated

import typer
import uvicorn
from circuit_supernodes.runtime.modal import SubjectConfig
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from modal_app.backend.subject import METHODS, SubjectBackend


def build_app(backend: SubjectBackend) -> FastAPI:
    app = FastAPI(title="circuit-supernodes subject")
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "methods": list(METHODS)}

    @app.post("/{method}")
    def call(method: str, args: Annotated[list, Body()]) -> JSONResponse:
        if method not in METHODS:
            raise HTTPException(404, f"unknown method {method!r}")
        with lock:
            return JSONResponse(getattr(backend, method)(*args))

    return app


def main(
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = SubjectConfig.model_fields["model_name"].default,
    transcoder_set: str = SubjectConfig.model_fields["transcoder_set"].default,
    node_threshold: float = SubjectConfig.model_fields["node_threshold"].default,
    edge_threshold: float = SubjectConfig.model_fields["edge_threshold"].default,
) -> None:
    backend = SubjectBackend(model_name, transcoder_set, node_threshold, edge_threshold)
    uvicorn.run(build_app(backend), host=host, port=port, timeout_keep_alive=3600)


if __name__ == "__main__":
    typer.run(main)
