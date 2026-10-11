# Bio-Circuit: causal circuit environments

Three text-only Verifiers v1 tasksets for interpreting Qwen3-4B through causal
feature interventions. This development release provides public evaluation
cohorts and source code. **Scored rollouts require an external CUDA GPU backend.**
The evaluation client runs on CPU; live interventions run on the GPU backend.

| Taskset ID | Task |
|---|---|
| `circuit-supernodes` | Infer a concept from evidence and identify causally necessary features |
| `supernode-rediscovery` | Rediscover a circuit for a named semantic goal |
| `suppressed-features` | Restore a subject model's output after feature corruption |

All three tasksets and the `circuit-null` harness ship in this one Python wheel.
The wheel includes backend source under `modal_app`; GPU dependencies are
installed separately on the GPU host.

## Public data

[amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit)
contains 16 evaluation tasks per taskset, full attribution graphs, provenance
manifests, SHA-256 hashes, schemas, and lightweight browsable indexes. The data
is CC-BY-4.0 and downloads without an HF token. The loader pins an immutable
dataset commit; a supplied local `dataset_file` takes precedence.

Compressed data totals about 2.6 GiB, downloaded one taskset at a time. Parsed
graphs need substantially more RAM. No training data or validated SFT
trajectories are included. These evaluation cohorts are not a disjoint train set.

## Install the environment client (no CUDA required)

Use Python 3.13 for the evaluation client:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install prime
prime env install wazupsteve/circuit-supernodes
```

The rediscovery/restoration modules and harness are included in that install.
This package uses the Verifiers v1 `Taskset` interface with `verifiers>=0.3.1,<0.4`.

## Modal backend

The environment calls `circuit-supernodes/Attributor` in the Modal workspace
selected by your credentials. That service loads Qwen3-4B and the transcoders on
A100-80GB GPUs. Leave `SUBJECT_URL` unset to use Modal.

For automated runners, configure `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` as
secrets available to the tool and reward processes. For a local client, use
`modal setup` to configure your profile.

If you need to deploy the service in your own workspace, extract the public
source archive from Prime Hub and run from its root:

```bash
modal deploy modal_app/attribution.py
```

The deployment defines its own Python 3.11 GPU image and persistent model cache.
It can scale to eight A100-80GB containers with a one-hour call timeout. Modal
bills GPU usage to the deployment's workspace. Package installation does not
create a deployment or configure credentials.

An existing self-hosted backend can also be selected through `SUBJECT_URL`.
It must implement the methods in `modal_app/server.py` and be reachable from
both tools and grading. That HTTP server has no built-in authentication; keep
it on loopback or a trusted private network.

## Run one evaluation

Save this as `eval.toml` and provide policy inference credentials through the
provider configuration supported by Verifiers. Inference and GPU hosting incur
costs.

```toml
model = "Qwen/Qwen3.5-4B"
num_tasks = 1
num_rollouts = 1

[sampling]
max_tokens = 16384

[env.taskset]
id = "circuit-supernodes"

[env.agent]
max_turns = 8

[env.agent.harness]
id = "circuit-null"
tool_timeout = 3600
```

```bash
eval @ eval.toml --dry-run --no-push
eval @ eval.toml --no-push
```

Change `env.taskset.id` to `supernode-rediscovery` or `suppressed-features` for the
other tasks. Defaults select the corresponding public evaluation file.
`--dry-run` checks configuration only; a real rollout establishes end-to-end
backend operation. `--no-push` prevents result uploads, not provider calls.

Keep heldout/control cohorts and corruption specifications inaccessible to the
policy. A public operator dataset is not a policy-visible tool response. These
tasks have no gold-circuit labels; successful SFT trajectories must be generated
and validated separately with live interventions and reward checks.

## Development

From `environments/circuit_supernodes` in a local checkout:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
uv build
```

The repository's `configs/rl/` files reference training data not included in this
release. Hosted training is not validated by local dry runs. Provision the
backend and supply separately generated training data before attempting RL.
