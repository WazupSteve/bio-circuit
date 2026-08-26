# `circuit-supernodes` package

This nested project builds and tests the three Verifiers-v1 TaskSet packages and owns the Modal backend and dataset builders. Conceptual documentation lives in the [repository docs](../../docs/README.md).

From this directory:

```bash
uv sync --frozen
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv build
```

Local config validation (no rollout or upload):

```bash
uv run eval @ ../../configs/eval/visible-dataset.toml --dry-run --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --dry-run --no-push
uv run eval @ ../../configs/eval/suppressed-features.toml --dry-run --no-push
uv run eval @ ../../configs/eval/suppressed-features-canary-egress.toml --dry-run --no-push
```

Remote deployment and builders incur GPU/API costs; see [operations](../../docs/operations.md). Dataset filenames and package interfaces are in [reference](../../docs/reference.md).
