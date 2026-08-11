# Bio-circuit environments

Three independent Verifiers 0.2.1 tasksets share graph, serialization,
validation, and retry primitives without sharing task contracts:

| Task | Plugin/taskset ID | Subject backend |
| --- | --- | --- |
| Causal circuit MDL | `circuit-supernodes` | Qwen attribution |
| Promptless discovery | `supernode-rediscovery` | Gemma discovery |
| Persistent output restoration | `output-restoration` | Qwen restoration |

Each task has dedicated eval/RL configs, a dataset namespace, a builder, and a
Modal app/class pair. See
[the environment guide](environments/circuit_supernodes/README.md) for exact
clean-checkout generation, manifest, validation, and deployment commands.

## Local setup

```bash
cd environments/circuit_supernodes
uv sync --group dev
uv run pytest
uv run ruff check .
```

The compact promptless rediscovery library and its manifest are checked in.
Causal MDL and output-restoration datasets contain generated attribution graphs,
are ignored by Git, and must be rebuilt as documented.

Read [FINDINGS.md](environments/circuit_supernodes/FINDINGS.md) before training
the causal task; it records the reward audit and accepted limitations.
