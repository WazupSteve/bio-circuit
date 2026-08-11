# Integrated circuit environments

This package exports three independent Verifiers tasksets. They reuse the
`circuit_supernodes` implementation package but retain separate plugin IDs,
configuration files, datasets, builders, and GPU services.

## Public surfaces

| Environment | Plugin/taskset ID | Eval config | RL config |
| --- | --- | --- | --- |
| Causal circuit MDL | `circuit-supernodes` | `configs/eval/circuit-mdl.toml` | `configs/rl/circuit-mdl.toml` |
| Promptless discovery | `supernode-rediscovery` | `configs/eval/supernode-rediscovery.toml` | `configs/rl/supernode-rediscovery.toml` |
| Output restoration | `output-restoration` | `configs/eval/output-restoration.toml` | `configs/rl/output-restoration.toml` |

The three Modal pairs are also independent:

| Purpose | App | Class | Deploy module |
| --- | --- | --- | --- |
| Qwen attribution and response banks | `circuit-supernodes-attribution` | `Attributor` | `modal_app/attribution.py` |
| Gemma promptless discovery | `circuit-supernodes-gemma-attribution` | `GemmaAttributor` | `modal_app/attribution_gemma.py` |
| Qwen persistent restoration | `circuit-supernodes-persistent-restoration` | `RestorationAttributor` | `modal_app/attribution_restoration.py` |

## Shared graph contract

The complete Circuit Tracer export is canonical task data. `GraphNode.activation`
and labels are optional when loading old slim graphs. Serialization preserves
activation, labels, all exported nodes, and all retained edges. `slim_graph`
creates an importance-ranked presentation view and can explicitly retain required
nodes for export/compatibility callers. Policy-selected views do not pass that
option, so they never receive hidden or corrupted nodes merely because those
nodes are required by a grader.

Label-free rendering powers promptless discovery. Causal MDL and restoration
store the complete graph and expose ordinary top-k views. Final and trial
restoration interventions are validated against the complete exported feature
universe, not a 60-node load-time slice.

## Task contracts

### Causal circuit MDL

The agent selects top-k graph views, runs bounded zero-ablation experiments, and
submits disjoint semantic supernodes plus a DAG. The fixed response bank covers
every exported feature. The answer-aware generator keeps
`CircuitSample.possible_answers`, samples within a bounded response window, and
records sampling disposition/provenance in retained rows and manifests.

The accepted model-node-count operating envelope is unchanged: an approximately
1,400-feature export produces a full response bank with `1 + 4N` subject passes
(`5,601` at `N=1,400`), and bank payloads remain attached once per dataset row.
Eval/RL row counts also remain the configured 16-task evaluation and 48/72-row
generation targets unless explicitly overridden.

### Promptless supernode discovery

The agent receives a curator description but no prompt or initial graph. Up to
five agent-authored prompts are traced through Gemma-2-2B. Returned graphs omit
human labels, and stable `layer_feature` identities are scored by hidden causal
replacement distance. The five-task curated library is compact and checked in.

### Persistent output restoration

The subject is replayed with a hidden persistent multiplicative corruption over
40–60 clean top-ranked features. The agent inspects ordinary top-k views and may
trial weighted interventions. A final proposal scores zero until it restores the
clean top token, then earns `exp(-KL(clean || restored))`.

Generated rows store the complete corrupted attribution graph. Corrupted nodes
remain in that universe so exact recovery is expressible, but no view forces all
of them into its top-k slice. Admission still screens the five accepted blanket
inverse baselines.

## Clean-checkout setup and deployment

Run from this directory:

```bash
uv sync --group dev
uv run modal setup
uv run modal deploy modal_app/attribution.py
uv run modal deploy modal_app/attribution_gemma.py
uv run modal deploy modal_app/attribution_restoration.py
```

Deploy commands mutate external Modal state. Run them only with the intended
Modal workspace selected. Local unit tests do not require a deployment.

## Dataset reproduction and manifests

Generated datasets live in separate ignored namespaces. Every builder writes a
sidecar `*.manifest.json` with row count, content digest, and generation
provenance. The response-bank precomputation step writes a new manifest for its
final output.

### Causal MDL (generated, ignored)

```bash
uv run python scripts/build_mdl_dataset.py --n 48 --split train \
  --neuronpedia-labels \
  --out circuit_supernodes/data/circuit_mdl/train.jsonl
uv run python scripts/build_mdl_dataset.py --n 72 --seed 1 --split eval \
  --neuronpedia-labels \
  --out circuit_supernodes/data/circuit_mdl/heldout.jsonl
uv run python scripts/precompute_response_banks.py \
  --input train.jsonl --out train-mdl.jsonl
uv run python scripts/precompute_response_banks.py \
  --input heldout.jsonl --out heldout-mdl.jsonl
```

The configs consume `train-mdl.jsonl` and `heldout-mdl.jsonl` under
`circuit_supernodes/data/circuit_mdl/`.

### Promptless discovery (compact, checked in)

```bash
uv run python scripts/build_rediscovery_dataset.py \
  --source https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmao6p1xd0000ypedw0qhlhob/gemma-fact-dallas-austin.json \
  --out circuit_supernodes/data/supernode_rediscovery/rediscovery.jsonl \
  --overwrite
```

Both `rediscovery.jsonl` and `rediscovery.manifest.json` are versioned.

### Output restoration (generated, ignored)

```bash
uv run python scripts/build_restoration_dataset.py --n 48 --seed 0 \
  --neuronpedia-labels \
  --out circuit_supernodes/data/output_restoration/output-restoration-train.jsonl
uv run python scripts/build_restoration_dataset.py --n 72 --seed 1 \
  --neuronpedia-labels \
  --out circuit_supernodes/data/output_restoration/output-restoration.jsonl
```

## Evaluation, training, and validation

```bash
uv run eval @ ../../configs/eval/circuit-mdl.toml --no-push
uv run eval @ ../../configs/eval/supernode-rediscovery.toml --no-push
uv run eval @ ../../configs/eval/output-restoration.toml --no-push

uv run pytest
uv run ruff check .
```

Hosted restoration canaries use
`configs/eval/output-restoration-canary-egress.toml` and
`configs/rl/output-restoration-canary-error.toml`; neither references an MDL
dataset. A full GPU smoke requires one task through each deployed backend and,
for restoration, one trial plus a final score.

## Layout

- `circuit_supernodes/`: shared graph code and task implementations
- `supernode_rediscovery/`: promptless plugin entrypoint
- `output_restoration/`: restoration plugin entrypoint
- `scripts/build_mdl_dataset.py`: answer-aware MDL builder
- `scripts/build_rediscovery_dataset.py`: compact curator-source builder
- `scripts/build_restoration_dataset.py`: persistent-corruption builder
- `tests/test_integration_contract.py`: combined schema and restoration-view regressions
