# Quickstart

Run one scored evaluation of each taskset. Commands run from
`environments/circuit_supernodes` unless noted.

## 1. Install

```bash
cd environments/circuit_supernodes
uv sync
```

Evaluation data downloads anonymously from Hugging Face on first use, one
taskset at a time (about 2.6 GiB compressed in total). No token is needed.

## 2. Log in to Prime

```bash
uv run prime login
```

The supplied configs use Prime Inference for the policy model and run the
harness in a Prime sandbox. The stored login covers both; `PRIME_API_KEY` in the
environment works too.

## 3. Connect the subject backend

```bash
uv run modal setup
```

Use credentials for the Modal workspace that hosts the `circuit-supernodes` app.
If that workspace has no deployment yet:

```bash
uv run modal deploy modal_app/attribution.py
```

The service runs Qwen3-4B on A100-80GB GPUs, up to eight containers, billed to
the workspace. Containers start on the first call (several minutes to load the
model) and stop after ten idle minutes.

Leave `SUBJECT_URL` unset to use Modal; set it only for a self-hosted GPU
server (`scripts/subject.sh up` starts one).

## 4. Check the configs

```bash
uv run eval @ ../../configs/eval/visible-dataset.toml --dry-run --no-push
```

A dry run validates and prints the resolved config without calling the policy
or the GPU.

## 5. Run a smoke test

```bash
uv run eval @ ../../configs/eval/visible-dataset.toml --num-tasks 1 --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --num-tasks 1 --no-push
uv run eval @ ../../configs/eval/suppressed-features.toml --num-tasks 1 --no-push
```

One task takes roughly 5–30 minutes depending on the policy model. Results go to
`outputs/<run>/traces.jsonl`; each trace records rewards, metrics and the
grader's per-prompt results under `info`.

Drop `--num-tasks 1` to run all 16 tasks, and pass `--model <id>` to try another
policy (list them with `uv run prime inference models`). `--no-push` keeps
results local; it does not stop inference or GPU charges.

## 6. Clean up

Containers scale down on their own. To stop them immediately:

```bash
uv run modal app stop circuit-supernodes   # also undeploys the app
```
