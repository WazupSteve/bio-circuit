# circuit-supernodes

Three Verifiers tasksets for finding causal circuits in Qwen3-4B with feature
interventions. The client runs on CPU; attribution and interventions run on a
GPU subject backend (Modal by default).

| Taskset | Task |
|---|---|
| `circuit-supernodes` | Infer a concept from five examples and find features necessary for it |
| `supernode-rediscovery` | Find features for a named goal, with no examples |
| `suppressed-features` | Undo a hidden feature corruption in the subject |

This package contains all three tasksets, the `circuit-null` harness and the
GPU backend source (`modal_app/`). It targets Verifiers `>=0.3.1,<0.4`.

## Data

Evaluation cohorts (16 tasks per taskset) download anonymously from
[amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit)
at a pinned revision, about 2.6 GiB compressed in total. Large files load
lazily, one row at a time. A local `dataset_file` takes precedence. No training
data is released.

## Install

From a checkout:

```bash
uv sync
```

The Prime Hub package (`prime env install wazupsteve/circuit-supernodes`) is
release 0.1.1, built on Verifiers 0.2.1; it doesn't include this version yet.

## Backend

With Modal credentials (`uv run modal setup`), clients call
`circuit-supernodes/Attributor` in your workspace. To deploy it:

```bash
uv run modal deploy modal_app/attribution.py
```

It runs on A100-80GB GPUs (up to eight containers, one-hour call timeout),
billed to the workspace. To use your own GPU server instead, set `SUBJECT_URL`;
it must serve the methods in `modal_app/server.py` and has no authentication.

## Run one evaluation

`eval.toml`:

```toml
model = "Qwen/Qwen3.5-4B"
num_tasks = 1
num_rollouts = 1

[sampling]
max_tokens = 16384

[env.taskset]
id = "circuit-supernodes"

[env.agent]
max_turns = 50

[env.agent.harness]
id = "circuit-null"
tool_timeout = 3600
```

```bash
uv run prime login                          # policy inference and sandbox
uv run eval @ eval.toml --dry-run --no-push # validate only
uv run eval @ eval.toml --no-push
```

Set `env.taskset.id` to `supernode-rediscovery` or `suppressed-features` for the
other tasks. Runs cost inference and GPU time; `--no-push` only keeps results
local.

Keep hidden cohorts and corruption data away from the policy. The tasks have no
gold circuits.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check .
uv build
```

Tests mock the backend and never start GPU work. The RL configs in
`configs/rl/` need training data that isn't released.
