# Module map

```text
configs/
  eval/                     Verifiers eval configs (one per taskset, plus a canary)
  rl/                       Training templates
environments/circuit_supernodes/          Core package `circuit-supernodes`
  circuit_supernodes/
    tasksets/               visible, rediscovery, restoration, shared base
    grading/                Ablation and restoration graders
    runtime/                Tools, graph store, parsing, datasets, subject clients
    models.py               Pydantic schemas (answers, rows, graphs)
  circuit_null/             `circuit-null` harness
  supernode_rediscovery/    Import entry point for that taskset ID
  suppressed_features/      Import entry point for that taskset ID
  modal_app/                Modal app, HTTP server, GPU backend (not checked by ty)
  scripts/                  Prompt generation and dataset builders
  tests/                    Unit and contract tests
environments/{circuit_null,supernode_rediscovery,suppressed_features}/
                            Thin Prime Hub distributions pinning the core package
scripts/                    subject.sh (self-hosted GPU), prepare_hf_dataset.py
docs/                       This documentation
```

## How the pieces connect

- **Tasksets** turn dataset rows into tasks and pick a grader.
- **Tools** (`runtime/circuit_tracer.py`) run in a tool server that Verifiers
  launches per rollout; the server rebuilds the task from its serialized data,
  so fields it needs (`subject_config`, `corruption_spec`) are serialized.
- **Subject clients** (`runtime/modal.py`) call the Modal `Attributor`, or an
  HTTP server when `SUBJECT_URL` is set.
- **GPU backend** (`modal_app/backend/`) loads Qwen3-4B with TransformerLens
  and serves sampling, attribution, ablation and restoration.

## Packaging

The root project provides development tools and Prime/Modal CLIs. The nested
`circuit-supernodes` project builds the wheel with all tasksets, the harness and
the backend source. The three Hub distributions only make the other IDs
installable.

## Tests

From the repository root or `environments/circuit_supernodes`:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uvx ty check
```

Tests cover schemas, graph views, parsing, interventions, builders, lazy and
eager loading, reward contracts, config layout and dataset resolution. Backend
calls are mocked; nothing deploys Modal or runs a live rollout.
