# BioCircuit

BioCircuit tests whether a language model can identify causal circuits in
Qwen3-4B. The policy inspects attribution graphs, runs feature interventions,
and submits a circuit. Grading checks its effects on held-out prompts and
controls.

## Tasks

| Taskset | Policy input | Grading |
|---|---|---|
| `circuit-supernodes` | Five labeled examples | Change the subject's output on positive examples while preserving controls |
| `supernode-rediscovery` | A semantic goal | Find features that affect a hidden positive cohort while preserving controls |
| `suppressed-features` | A corrupted task family | Restore the subject's clean output distribution |

All three tasksets use the shared `circuit-null` harness. The policy runs through
Verifiers; Qwen3-4B and its transcoders run on a Modal A100-80GB backend. The
client does not need a local GPU. Live experiments and grading use Modal GPU
compute.

## Public release

- [Dataset on Hugging Face](https://huggingface.co/datasets/amitprakash2005/bio-circuit):
  48 evaluation tasks, full graphs, browsable indexes, schemas, and provenance.
- [Circuit Supernodes](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-supernodes)
- [Supernode Rediscovery](https://app.primeintellect.ai/dashboard/environments/wazupsteve/supernode-rediscovery)
- [Suppressed Features](https://app.primeintellect.ai/dashboard/environments/wazupsteve/suppressed-features)
- [Shared harness](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-null)

The environment downloads a pinned dataset revision on first use. Each taskset
has 16 evaluation tasks. The generated data is CC-BY-4.0; this release does not
include training splits or validated SFT trajectories.

## Run

Follow the [quickstart](quickstart.md) to install the client, connect to the
Modal backend, and run an evaluation. You need Python 3.13, `uv`, access to the
Modal deployment, and policy inference credentials.

To install the published package in an active virtualenv:

```bash
uv pip install prime
prime env install wazupsteve/circuit-supernodes
```

That package includes all three tasksets and the harness. The other Hub entries
allow each component to be selected by its own ID.

## Development

From the repository root:

```bash
cd environments/circuit_supernodes
uv sync --frozen
uv run pytest
uv run ruff check .
uv build
```

Tests use local fixtures and mocked backend calls. They do not start GPU jobs.
Datasets, rollout traces, logs, and build outputs are excluded from Git.

See [the documentation](docs/README.md) for tool contracts, grading, dataset
builders, and deployment details.
