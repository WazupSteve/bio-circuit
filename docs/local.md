# Self-hosted subject

The subject (Qwen3-4B plus transcoders) can run on any single 80 GB GPU instead
of Modal. Clients choose the backend with one variable: if `SUBJECT_URL` is set
they call that HTTP server, otherwise Modal. Nothing else changes.

## Quick start

From the repository root:

```bash
scripts/subject.sh up                      # first run ~10 min (~57 GB of weights), later ~3 min
export SUBJECT_URL=http://localhost:8000
(cd environments/circuit_supernodes && uv run eval @ ../../configs/eval/visible-dataset.toml --num-tasks 1 --no-push)
scripts/subject.sh down                    # stop the pod; it bills hourly
```

`up` creates the cheapest A100-80GB pod on Prime Intellect (run
`uv run prime login` first), copies `environments/circuit_supernodes` to it,
installs the pinned GPU stack (Python 3.11, `torch==2.8.0`,
`transformer-lens==2.16.1`, `pyarrow<26`, …), starts `modal_app.server`, opens an
SSH tunnel to `localhost:8000` and waits for `/health`.

To use a GPU host you can already SSH into:

```bash
scripts/subject.sh up ubuntu@my-gpu-host
```

| Variable | Default | Purpose |
|---|---|---|
| `SUBJECT_GPU` | `A100_80GB` | Pod GPU type |
| `SUBJECT_PORT` | `8000` | Local tunnel port |
| `SUBJECT_SSH_KEY` | `~/.ssh/primeintellect_ed25519` | SSH key |
| `SUBJECT_INSTANCE` | — | Names a pod for parallel runs; stop it with the same value |

For parallel evaluations give each pod its own `SUBJECT_INSTANCE` and
`SUBJECT_PORT`.

## What runs where

| Component | Where | Role |
|---|---|---|
| `modal_app/backend/subject.py` | GPU host | `SubjectBackend`: loads the model once and serves the RPCs; no Modal dependency |
| `modal_app/server.py` | GPU host | FastAPI: `GET /health`, `POST /<method>` with a JSON list of arguments; one lock serializes GPU work |
| `HttpSubject` (`circuit_supernodes/runtime/modal.py`) | Your machine | Same `call` / `map` / `spawn` interface as the Modal client |

The tools and graders call the subject from your machine, so a `localhost`
tunnel works with the default configs. The server has no authentication; keep
it on loopback or a trusted network.

Server log: `~/subject.log` on the host. Restart by hand with
`cd ~/bio-circuit/circuit_supernodes && .venv/bin/python -m modal_app.server --port 8000`.

## Expectations

- A short prompt attributes in about ten seconds on an A100 once the model is
  loaded; long prompts take longer. Interventions take seconds.
- One GPU runs `map` calls one after another, where Modal fans out to eight
  containers. For bulk generation, run several hosts or one server per GPU.
- Weights cache on the pod disk; a new pod downloads them again.
