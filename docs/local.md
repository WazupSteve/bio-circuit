# Local / self-hosted subject (no Modal)

The subject model (Qwen3-4B + transcoders) can run on any single 80 GB-class GPU instead of Modal. The environment picks the backend from one env var: if `SUBJECT_URL` is set it talks HTTP to that server, otherwise it uses the Modal app. Nothing else changes.

## One command

```bash
scripts/subject.sh up        # ~10 min first time (downloads ~57 GB of weights), ~3 min after
export SUBJECT_URL=http://localhost:8000
cd environments/circuit_supernodes && uv run eval @ ../../configs/eval/visible-dataset.toml --no-push
scripts/subject.sh down      # terminate the pod when finished — it bills hourly
```

The configs use the local `circuit-null` harness, which keeps the isolated MCP
client on the MCP 1.x version required by Verifiers.

`up` creates the cheapest available A100-80GB pod on Prime Intellect (`uv run prime login` first), rsyncs `environments/circuit_supernodes` to it, installs the pinned GPU stack into a Python 3.11 venv, starts `modal_app.server`, opens an SSH tunnel to `localhost:8000`, and waits for `/health`.

Already have a GPU box you can SSH into? Skip Prime entirely:

```bash
scripts/subject.sh up ubuntu@my-gpu-host
```

Knobs (env vars): `SUBJECT_GPU` (default `A100_80GB`), `SUBJECT_PORT` (`8000`), `SUBJECT_SSH_KEY` (`~/.ssh/primeintellect_ed25519`).

For parallel evaluations, set a unique `SUBJECT_INSTANCE` and `SUBJECT_PORT`
for each pod. The instance name namespaces the local pod and tunnel state, so
each instance must be stopped with the same `SUBJECT_INSTANCE` value.

## What is running

| Piece | Where | Purpose |
|---|---|---|
| `modal_app/backend/subject.py` | GPU host | `SubjectBackend` — loads the model once, serves the five operations. No Modal dependency. |
| `modal_app/server.py` | GPU host | FastAPI: `GET /health`, `POST /<method>` with a JSON list of positional args. One lock serialises GPU work. |
| `HttpSubject` in `circuit_supernodes/runtime/modal.py` | your machine | Same `call` / `map` / `spawn` surface as `ModalSubject`. |

Server log on the host: `~/subject.log`. Restart manually with
`cd ~/bio-circuit/circuit_supernodes && .venv/bin/python -m modal_app.server --port 8000`.

## Expectations

- Attribution takes ~2–4 min per graph on an A100; sampling and interventions take seconds.
- One GPU means `map` calls run sequentially (Modal fanned out to 8 containers). For bulk dataset generation, bring up several hosts and shard prompts, or run one server per GPU on a multi-GPU box.
- The HF cache lives on the pod disk; terminating the pod re-downloads weights next time.
