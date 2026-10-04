# Module map

```text
configs/
  eval/                         Local Verifiers evaluation configs
  rl/                           Training templates
environments/circuit_supernodes/
  circuit_supernodes/
    grading/                    Ablation and restoration rewards
    runtime/                    Data loading, graph tools, subject clients
    tasksets/                   Three Verifiers tasksets
  circuit_null/                 Shared policy harness
  supernode_rediscovery/         Rediscovery import entrypoint
  suppressed_features/          Restoration import entrypoint
  modal_app/                    Modal deployment and GPU backend
  scripts/                      Prompt and dataset builders
  tests/                        Unit and integration contract tests
environments/circuit_null/      Hub distribution for the harness
environments/supernode_rediscovery/  Hub distribution for rediscovery
environments/suppressed_features/    Hub distribution for restoration
scripts/                       Optional GPU-host helper and release preparation
docs/                          Setup and reference documentation
```

## Execution

`modal_app/backend/subject.py` loads Qwen3-4B and the transcoders. The Modal
service exposes sampling, attribution, corruption, ablation, and restoration
methods. `circuit_supernodes/runtime/modal.py` calls those methods through the
Modal SDK, or uses an HTTP backend when `SUBJECT_URL` is set.

Tasksets shape policy inputs and select graders. The shared tools expose
attribution, paginated graph queries, and feature interventions. Hidden cohorts
and corruption parameters stay in task data used by the backend and graders.

## Packaging

The root Python project supplies Prime and development tools. The nested
`circuit-supernodes` project builds the core wheel, including all tasksets,
the harness, and backend source. Three small Hub distributions depend on that
core package and make the other tasksets and harness addressable by Hub ID.

Datasets live on Hugging Face. The loader uses a pinned release revision when
no local dataset is supplied. Generated data, logs, traces, and build outputs
are ignored by Git.

## Tests

Tests cover schemas, graph views, response parsing, interventions, dataset
builders, taskset loading, reward contracts, and public dataset resolution.
Remote calls are mocked; the tests do not deploy Modal or verify a live rollout.

From `environments/circuit_supernodes`:

```bash
uv sync --frozen
uv run pytest
uv run ruff check .
```
