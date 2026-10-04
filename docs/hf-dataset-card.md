---
language:
- en
license: cc-by-4.0
pretty_name: Bio-Circuit causal interpretability evaluation cohorts
size_categories:
- n<1K
tags:
- interpretability
- causal-interventions
- reinforcement-learning
- text-only
- requires-gpu
configs:
- config_name: visible
  data_files:
  - split: test
    path: indexes/visible.jsonl
- config_name: rediscovery
  data_files:
  - split: test
    path: indexes/rediscovery.jsonl
- config_name: restoration
  data_files:
  - split: test
    path: indexes/restoration.jsonl
---

# Bio-Circuit

Public evaluation cohorts for three text-only, causal circuit interpretability
tasks on Qwen3-4B. The release contains 48 tasks (16 per taskset), full attribution
graphs, hidden evaluation cohorts, generation manifests, and SHA-256 checksums.
There is no training split and no validated SFT trajectory collection in this
release. Do not describe these 48 evaluation tasks as independent training data.

The dataset is downloadable without credentials. **Live tool calls and reward
computation require a CUDA GPU subject backend** (the existing setup uses an
A100-80GB). Precomputed graphs do not replace live causal interventions. This
release is not eligible for a run that excludes GPU-dependent environments.

## Tasksets

| Config | Prime environment | Task | Test rows |
|---|---|---|---|
| `visible` | `wazupsteve/circuit-supernodes` | Infer a concept from evidence and identify causally necessary, specific features | 16 |
| `rediscovery` | `wazupsteve/supernode-rediscovery` | Rediscover features for a named semantic goal | 16 |
| `restoration` | `wazupsteve/suppressed-features` | Identify interventions that restore a corrupted subject's output | 16 |

All tasksets use the `circuit-null` harness included in the environment package.
The environment package, source archive, and setup instructions are available
through Prime's Environments Hub.

## Download and inspect

The Dataset Viewer and `load_dataset` expose **lightweight task indexes**. Each
index row points to a full graph-bearing cohort in a compressed JSONL file;
index rows alone are not runnable tasks or labeled model responses.

```python
from datasets import load_dataset

tasks = load_dataset("amitprakash2005/bio-circuit", "visible", split="test")
print(tasks[0])
```

Download a full cohort file without authentication:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "amitprakash2005/bio-circuit",
    "data/visible_dataset/heldout-cohorts.jsonl.gz",
    repo_type="dataset",
    token=False,
    # For reproducibility, supply revision=<release commit SHA>.
)
```

The environment pins an immutable dataset commit and downloads its own taskset's
file on first use. Compressed files total about 2.6 GiB; decompressed graphs and
parsed task objects require considerably more memory. Read one cohort at a time
with `gzip.open(path, "rt")`; do not load all graph-bearing files into memory.

## Files and schema

- `indexes/*.jsonl`: task ID, family, goal, cohort counts, full data filename,
  zero-based row index, and SHA-256 of the compressed data file.
- `data/visible_dataset/heldout-cohorts.jsonl.gz`: visible evidence tasks.
- `data/supernode_rediscovery/promptless-supernodes.jsonl.gz`: rediscovery tasks.
- `data/output_restoration/suppressed-features.jsonl.gz`: restoration tasks.
- `data/**/*.manifest.json`: original generation provenance sidecars.
- `inventory.json`: verified file sizes, row counts, and compressed-file hashes.
- `schema.json`: the environment's JSON Schema for a full cohort row.

Each full row has `family`, optional `goal`, and three lists: `example_prompts`,
`heldout_prompts`, and `control_prompts`. Each example has `prompt`,
`target_token`, `graph`, and optional `corruption`. Graphs contain metadata,
nodes, and links. Corruption entries are `(layer, feature, position, factor)`.
Stable intervention identities are `layer_feature`; graph nodes can also carry
a token position. The environment's schema and parser define the exact types.

Original manifests are retained verbatim. In particular, the visible manifest's
`dataset` field uses the historical name `heldout-cohorts-v2.jsonl.gz`; the
published path is `heldout-cohorts.jsonl.gz`. Its checksum refers to the published
bytes. Loopback backend URLs in manifests describe the generation setup, not a
public endpoint.

## Construction and validation

Subject model: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
Transcoders: [mwhanna/qwen3-4b-transcoders](https://huggingface.co/mwhanna/qwen3-4b-transcoders).
Graphs use Circuit Tracer with node threshold 0.8 and edge threshold 0.98.
The manifests record sampling seeds, cohort construction, screening, generation
policy, schema version, and aggregate graph-quality measurements.

Task families include two-hop reasoning, agreement, comparison, weekdays, path
composition, modular arithmetic, rule composition, spatial composition, and
Boolean composition. Every released row is checked against the environment's
Pydantic schema, and row counts and hashes are checked against the original
manifests before upload. This is structural/provenance validation, not a new
GPU replay or a claim of successful agent performance.

## Evaluation boundaries and limitations

Heldout prompts, control prompts, and corruption specifications are public to
operators but must remain outside the evaluated policy's messages and tools.
Prevent policy access to raw dataset files when running a meaningful evaluation;
public availability is not a secrecy guarantee. The indexes also expose goals
that some tasksets intentionally withhold from the policy.

The 16 tasks per family are a small research evaluation set, not a representative
sample of all model behavior. Causal rewards depend on the subject model,
transcoders, numerical stack, and backend configuration. Generation and model
execution can be nondeterministic. Graph approximations and cohort selection
introduce limitations described by the generation manifests and source code.

This release contains task inputs and grading data, not verified gold circuits
or successful agent trajectories. SFT data generation needs policy rollouts,
live GPU-backed grading, and a separately documented acceptance procedure.

## License and attribution

The generated dataset is released under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Attribute Bio-Circuit and link to this dataset, noting modifications. Model
weights, transcoders, and third-party software are not included and retain their
respective licenses. Dataset licensing does not relicense those dependencies.
