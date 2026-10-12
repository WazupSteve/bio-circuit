# Troubleshooting

## Subject containers crash with `pyarrow requires NumPy 2.0`

The GPU image was built without the `pyarrow<26` pin: `transformer-lens` keeps
NumPy 1.x, and pyarrow 26 needs NumPy 2. Redeploy from this repository
(`uv run modal deploy modal_app/attribution.py`); the pin is in
`modal_app/backend/deployment.py` and `scripts/subject.sh`. Crashing containers
restart and bill GPU time, so stop them first.

## An eval runs but logs nothing for minutes

Normal on a cold start: the first call loads Qwen3-4B and the transcoders.
Check the backend with `uv run modal app logs circuit-supernodes`, and the run's
own log under `outputs/<run>/logs/`.

## Dataset download fails

Evaluation files download anonymously from the pinned Hugging Face revision.
Check network access and free disk space. A local `dataset_file` takes
precedence; if it is a stale Git LFS pointer or truncated, move it aside
(`gzip -t <file>` checks a gzip file). Training files are not published.

## Taskset or harness not found

```bash
uv sync
uv run python -c 'from verifiers.v1.utils.loaders import taskset_class; print(taskset_class("supernode-rediscovery"))'
```

IDs map to modules with underscores (`supernode-rediscovery` →
`supernode_rediscovery`). The package must be installed in the interpreter that
runs `eval`; the root project's venv does not include it, so run from
`environments/circuit_supernodes`.

## Modal credentials or missing app

`modal.Cls.from_name("circuit-supernodes", "Attributor")` must resolve in the
active workspace (`uv run modal profile current`). Deploy it there if needed,
and unset `SUBJECT_URL` unless you mean to use an HTTP backend. Hosted runners
need the Modal tokens in both tool and reward processes; keep them in runner
secrets, never in configs.

## Policy or sandbox errors

Policy inference and sandboxes both use your Prime login (`uv run prime login`,
or `PRIME_API_KEY`). Check that the model ID exists with
`uv run prime inference models`. Policy-provider errors are separate from
backend errors, which retry four times (1, 4 and 15 s waits).

## Rewards are all zero

Check the trace before blaming the policy. In `traces[0]`:

- `metrics.baseline_drift = 1` or `grading_validation_error = 1`: the replay did
  not reproduce the stored baseline (backend or numerical drift, or a tied
  answer in the data).
- `info.*_score.error` set: the final answer had no parseable JSON, often
  because the run hit `max_turns`.
- Otherwise the answer was graded and failed the thresholds; the score dict
  shows per-gate rates.

## Results don't match earlier runs

Compare the subject model, transcoders, Circuit Tracer version, pruning
thresholds and backend stack against the dataset manifest, and confirm the
grading thresholds match. Loaders don't enforce manifest settings.
