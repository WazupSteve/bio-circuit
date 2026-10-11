# Troubleshooting

## Dataset download fails

Default evaluation files download anonymously from the pinned Hugging Face
release. Check network access, free disk space, and the Hugging Face cache.
All three tasksets have published evaluation data; training files must be
created separately.

A local `dataset_file` takes precedence over the public download. If you have
an old Git LFS pointer at that path, move the pointer aside so the loader can
download the real file. Use `gzip -t <path>` to check a compressed local file.

## Taskset import fails

From `environments/circuit_supernodes`:

```bash
uv sync --frozen
uv run python -c 'from verifiers.v1.utils.loaders import taskset_class; print(taskset_class("supernode-rediscovery"))'
```

Taskset IDs map to Python modules with underscores. Install the core package
in the interpreter used by the runner; reinstall after packaging changes.

## Modal credentials or deployment

Confirm that your Modal credentials target the workspace containing the
`circuit-supernodes` app and its `Attributor` class. If you need your own
instance, deploy `modal_app/attribution.py` as described in the quickstart.
Unset `SUBJECT_URL` unless you intend to use an HTTP backend.

Hosted runners need access to the same credentials in both the tool and
reward processes. Keep tokens in runner secrets, not in TOML files or Git.

## Provider failures and retries

Policy-provider errors and Modal errors are separate. Modal grading calls retry
four times, waiting 1, 4, and 15 seconds between attempts. Inspect the original
error before increasing concurrency. Dataset builders record rejected samples
and graph failures in their generation summaries.

## Runtime selection

The supplied configs use `circuit-null` with local subprocess execution.
The subject model still runs on Modal. Selecting a Docker, Prime, or Modal
harness runtime changes where the policy program executes; it does not by
itself deploy the subject backend. An HTTP subject must be reachable from both
the tools and the grader; localhost inside a sandbox refers to that sandbox.

## Model drift

Check the subject model, transcoders, Circuit Tracer version, pruning thresholds,
and backend numerical stack against the dataset manifest. Runtime row validation
does not enforce those settings. Compare results only when their provenance
and grading configuration are compatible.
