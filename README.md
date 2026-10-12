# BioCircuit

BioCircuit asks a policy model to find causal circuits inside Qwen3-4B. The
policy attributes prompts, inspects attribution graphs, runs feature
interventions, and submits a circuit. A grader replays the submission on hidden
prompts and scores its causal effect.

| Taskset | The policy sees | The grader checks |
|---|---|---|
| `circuit-supernodes` | Five example prompts sharing a hidden concept | Negating the submitted supernode changes the answer on the examples and on 25 hidden same-concept prompts, and leaves 25 controls unchanged |
| `supernode-rediscovery` | Only a goal name, e.g. `comparison` | Same causal test on a hidden 4-prompt cohort and 4 controls |
| `suppressed-features` | Only a task family; the subject carries a hidden feature corruption | Submitted multipliers restore the clean output distribution on 10 hidden prompts |

All three tasksets share the `circuit-null` harness and run on
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) 0.3.1. Qwen3-4B and
its transcoders run on a Modal A100-80GB service, so the client needs no GPU.

## Status

The environment runs end to end: all three tasksets have been evaluated against
the live Modal subject. Rewards are sparse by design; the eval configs set
calibration thresholds ([tasksets](docs/tasksets.md)).

## Public release

- Dataset: [amitprakash2005/bio-circuit](https://huggingface.co/datasets/amitprakash2005/bio-circuit)
  (16 evaluation tasks per taskset, CC-BY-4.0, no training split).
- Prime Hub: [circuit-supernodes](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-supernodes),
  [supernode-rediscovery](https://app.primeintellect.ai/dashboard/environments/wazupsteve/supernode-rediscovery),
  [suppressed-features](https://app.primeintellect.ai/dashboard/environments/wazupsteve/suppressed-features),
  [circuit-null](https://app.primeintellect.ai/dashboard/environments/wazupsteve/circuit-null).

The Hub packages are release 0.2.1 (Verifiers 0.3.1). Install
`circuit-supernodes` first; the other three entries are thin aliases that depend
on it:

```bash
uvx --from prime prime env install wazupsteve/circuit-supernodes
```

## Get started

Follow the [quickstart](quickstart.md). You need Python 3.13, `uv`, a Prime
account (policy inference and sandboxes), and Modal access.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests use local fixtures and mocked backend calls; they never start GPU work.
Datasets, traces, logs and build outputs stay out of Git.

The [documentation](docs/README.md) covers tools, grading, datasets,
deployment and the source layout.
