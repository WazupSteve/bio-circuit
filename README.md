# Bio-circuit

v1 verifier compatible prime environment for supernode circuit identification on Qwen3-4B attribution graphs.

The environment lives in `environments/circuit_supernodes`.

## Quickstart

```bash
cd environments/circuit_supernodes
uv run python scripts/build_dataset.py
prime env install circuit-supernodes
prime eval run circuit-supernodes
```

Dataset generation uses Modal to compute attribution graphs. The generated dataset is ignored by git and can be rebuilt when needed.
